"""Banc de bout en bout : faux client WebSocket qui rejoue des WAV et mesure
fin de parole -> premier octet audio du prof. Le serveur doit tourner (run.bat).

Usage : python scripts/bench_e2e.py [dossier_wav] [--mode ptt|handsfree] [--limit 20]
Longue séance (fuites, contexte du LLM) : --turns 200 --long-every 20 rejoue les WAV en boucle, avec
de temps en temps un énoncé de ~30 s entrecoupé d'hésitations, et fait le point tous les 10 échanges.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from websockets.asyncio.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.bench_stt_verbatim import REC, load_16k  # noqa: E402
from server.config import load_config  # noqa: E402

FRAME = 512
BENCH_USER = "Banc d'essai"


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.first_audio: asyncio.Future | None = None
        self.done: asyncio.Future | None = None
        self.chunks: list[tuple[float, float]] = []  # (arrivée, durée en s)
        self.metrics: list[dict] = []  # latences de chaque tour, vues du serveur
        self.errors = 0

    def gaps(self) -> tuple[float, int]:
        """Lecture simulée : total des silences non voulus (lecteur à vide) et nombre de trous > 20 ms."""
        if not self.chunks:
            return 0.0, 0
        cursor, total, count = self.chunks[0][0], 0.0, 0
        for arrival, duration in self.chunks:
            if arrival > cursor:
                total += arrival - cursor
                count += arrival - cursor > 0.02
                cursor = arrival
            cursor += duration
        return total, count

    async def reader(self) -> None:
        async for msg in self.ws:
            if isinstance(msg, bytes):
                turn, seg = np.frombuffer(msg[:8], dtype="<u4")
                now = time.perf_counter()
                self.chunks.append((now, (len(msg) - 8) / 2 / 24000))
                if self.first_audio and not self.first_audio.done():
                    self.first_audio.set_result(now)
                await self.ws.send(json.dumps({"type": "seg_start", "turn": int(turn), "seg": int(seg)}))
                continue
            ev = json.loads(msg)
            if ev["type"] == "assistant_done":
                await self.ws.send(json.dumps({"type": "played", "turn": ev["turn"]}))
                if self.done and not self.done.done():
                    self.done.set_result(ev["turn"])
            elif ev["type"] == "noinput" and self.done and not self.done.done():
                self.done.set_result(None)
            elif ev["type"] == "metrics":
                self.metrics.append(ev["latency_ms"])
            elif ev["type"] == "error":
                self.errors += 1
                print(f"  ERREUR du serveur : {ev.get('message')}", flush=True)

    def arm(self) -> None:
        loop = asyncio.get_running_loop()
        self.first_audio, self.done = loop.create_future(), loop.create_future()
        self.chunks = []


def frames(audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    pcm = np.concatenate([pcm, np.zeros((-len(pcm)) % FRAME, "<i2")])
    for i in range(0, len(pcm), FRAME):
        yield pcm[i: i + FRAME].tobytes()


def trim(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Retire les silences du début et de la fin (on garde 30 ms)."""
    voiced = np.flatnonzero(np.abs(audio) > threshold)
    if not len(voiced):
        return audio
    return audio[max(0, voiced[0] - 480): voiced[-1] + 480]


def long_utterance(clips: list[np.ndarray], start: int, seconds: float) -> np.ndarray:
    """Phrases enchaînées avec des hésitations de 200 ms : assez pour lancer puis annuler des tours
    spéculatifs, trop peu pour que le prof prenne la parole (min_release_ms)."""
    gap = np.zeros(int(0.2 * 16000), np.float32)
    parts, i = [], start
    while sum(len(p) for p in parts) < seconds * 16000:
        parts += [trim(clips[i % len(clips)]), gap]
        i += 1
    return np.concatenate(parts[:-1])


class Resources:
    """VRAM de tout le GPU et mémoire du processus serveur, pour repérer une fuite."""

    def __init__(self):
        import psutil
        import pynvml

        pynvml.nvmlInit()
        self.nvml, self.gpu = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        # le vrai python (le plus gros), pas le petit lanceur du .venv qui porte la même ligne de commande
        self.server = max((p for p in psutil.process_iter(["cmdline", "memory_info"])
                           if "server.main" in " ".join(p.info["cmdline"] or [])),
                          key=lambda p: p.info["memory_info"].rss, default=None)

    def __str__(self) -> str:
        vram = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu).used / 2**30
        ram = f", RAM serveur {self.server.memory_info().rss / 2**30:.2f} Go" if self.server else ""
        return f"VRAM {vram:.2f} Go{ram}"


def bench_user(base: str) -> int:
    for u in httpx.get(f"{base}/api/users").json():
        if u["name"] == BENCH_USER:
            return u["id"]
    return httpx.post(f"{base}/api/users", json={"name": BENCH_USER}).json()["id"]


async def run(folder: Path, mode: str, limit: int, subject: str, turns: int | None, long_every: int,
              report_every: int) -> None:
    cfg = load_config()["server"]
    user = bench_user(f"http://{cfg['host']}:{cfg['port']}")
    wavs = sorted(folder.glob("*.wav"))[:limit]
    clips = [load_16k(path) for path in wavs]
    resources = Resources()
    print(f"Départ : {resources}", flush=True)
    latencies, gaps = [], []
    url = f"ws://{cfg['host']}:{cfg['port']}/ws?user={user}&subject={subject}"
    async with connect(url, max_size=2**22) as ws:
        client = Client(ws)
        reader = asyncio.create_task(client.reader())
        await ws.send(json.dumps({"type": "mode", "value": mode}))
        client.arm()
        await ws.send(json.dumps({"type": "start"}))
        await client.done  # salutation du prof
        silence = np.zeros(FRAME, "<i2").tobytes()
        for i in range(turns or len(wavs)):
            if long_every and (i + 1) % long_every == 0:
                name, audio = f"long{i + 1}", long_utterance(clips, i, 30)
            else:
                name, audio = wavs[i % len(wavs)].stem, clips[i % len(clips)]
            client.arm()
            if mode == "ptt":
                await ws.send(json.dumps({"type": "ptt", "state": "down"}))
            t_next = time.perf_counter()
            for f in frames(audio):  # temps réel, comme un micro
                await ws.send(f)
                t_next += FRAME / 16000
                await asyncio.sleep(max(0, t_next - time.perf_counter()))
            t_end = time.perf_counter()
            if mode == "ptt":
                await ws.send(json.dumps({"type": "ptt", "state": "up"}))
            else:
                for _ in range(int(2.5 * 16000 / FRAME)):
                    await ws.send(silence)
                    t_next += FRAME / 16000
                    await asyncio.sleep(max(0, t_next - time.perf_counter()))
            try:
                first = await asyncio.wait_for(client.first_audio, 20)
                latencies.append((first - t_end) * 1000)
                await asyncio.wait_for(client.done, 60)
                gap, n_gaps = client.gaps()
                gaps.append(gap * 1000)
                print(f"{i + 1:3d} {name} ({len(audio) / 16000:.0f} s) : premier son {latencies[-1]:.0f} ms, "
                      f"trous audio {gap * 1000:.0f} ms ({n_gaps} > 20 ms)", flush=True)
            except TimeoutError:
                print(f"{i + 1:3d} {name} : pas de réponse", flush=True)
            if report_every and (i + 1) % report_every == 0:
                recent = latencies[-report_every:] or [0]
                llm = [m.get("llm_done", 0) / 1000 for m in client.metrics[-report_every:]] or [0]
                print(f"--- {i + 1} échanges : {resources} | premier son p50 {statistics.median(recent):.0f} ms, "
                      f"max {max(recent):.0f} ms | réponse du LLM finie en {max(llm):.1f} s au plus | "
                      f"trous audio max {max(gaps[-report_every:] or [0]):.0f} ms | erreurs {client.errors}",
                      flush=True)
            await asyncio.sleep(0.3)
        reader.cancel()
    print(f"Arrivée : {resources}")
    if latencies:
        lat = sorted(latencies)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        print(f"\n[{mode}] fin de parole -> premier son : p50 {statistics.median(lat):.0f} ms, p95 {p95:.0f} ms "
              f"(n={len(lat)}) ; trous audio par réponse : médiane {statistics.median(gaps):.0f} ms, "
              f"max {max(gaps):.0f} ms ; erreurs du serveur : {client.errors}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(REC / "synth"))
    ap.add_argument("--mode", choices=["ptt", "handsfree"], default="ptt")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--subject", default="anglais")
    ap.add_argument("--turns", type=int, help="nombre d'échanges (les WAV sont rejoués en boucle)")
    ap.add_argument("--long-every", type=int, default=0, help="un énoncé de ~30 s tous les N échanges")
    ap.add_argument("--report-every", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(run(Path(args.folder), args.mode, args.limit, args.subject, args.turns, args.long_every,
                    args.report_every))
