"""Banc de bout en bout : faux client WebSocket qui rejoue des WAV et mesure
fin de parole -> premier octet audio du prof. Le serveur doit tourner (run.bat).

Usage : python scripts/bench_e2e.py [dossier_wav] [--mode ptt|handsfree] [--limit 20]
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

    def arm(self) -> None:
        loop = asyncio.get_running_loop()
        self.first_audio, self.done = loop.create_future(), loop.create_future()
        self.chunks = []


def frames(audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    pcm = np.concatenate([pcm, np.zeros((-len(pcm)) % FRAME, "<i2")])
    for i in range(0, len(pcm), FRAME):
        yield pcm[i: i + FRAME].tobytes()


def bench_user(base: str) -> int:
    for u in httpx.get(f"{base}/api/users").json():
        if u["name"] == BENCH_USER:
            return u["id"]
    return httpx.post(f"{base}/api/users", json={"name": BENCH_USER}).json()["id"]


async def run(folder: Path, mode: str, limit: int, subject: str) -> None:
    cfg = load_config()["server"]
    user = bench_user(f"http://{cfg['host']}:{cfg['port']}")
    wavs = sorted(folder.glob("*.wav"))[:limit]
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
        for path in wavs:
            audio = load_16k(path)
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
                print(f"{path.stem} : premier son {latencies[-1]:.0f} ms, trous audio {gap * 1000:.0f} ms "
                      f"({n_gaps} > 20 ms)", flush=True)
            except TimeoutError:
                print(f"{path.stem} : pas de réponse")
            await asyncio.sleep(0.3)
        reader.cancel()
    if latencies:
        lat = sorted(latencies)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        print(f"\n[{mode}] fin de parole -> premier son : p50 {statistics.median(lat):.0f} ms, p95 {p95:.0f} ms "
              f"(n={len(lat)}) ; trous audio par réponse : médiane {statistics.median(gaps):.0f} ms, "
              f"max {max(gaps):.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(REC / "synth"))
    ap.add_argument("--mode", choices=["ptt", "handsfree"], default="ptt")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--subject", default="anglais")
    args = ap.parse_args()
    asyncio.run(run(Path(args.folder), args.mode, args.limit, args.subject))
