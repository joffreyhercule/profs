"""Banc par composant : latence de chaque étage, puis VRAM avec tout chargé ensemble.

Usage : python scripts/bench_components.py [--runs 5]
"""

import argparse
import asyncio
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_config  # noqa: E402
from server.main import vram_used_gb  # noqa: E402
from server.subjects import load_subjects  # noqa: E402

TTS_TEXTS = [
    ("en", "Nice try, but we usually say I went to the cinema."),
    ("fr", "Presque ! On dit plutôt, je suis allée au cinéma."),
    ("en", "Great!"),
]


def med(xs: list[float]) -> str:
    return f"médiane {statistics.median(xs):.0f} ms (min {min(xs):.0f}, max {max(xs):.0f})"


def bench_tts(cfg: dict, runs: int):
    from server.tts import TeacherTTS, TTSJob

    t0 = time.perf_counter()
    subject = load_subjects()["anglais"]
    tts = TeacherTTS(cfg["tts"], {subject.id: subject.voice})
    tts.warmup()
    print(f"TTS chargé + préchauffé en {time.perf_counter() - t0:.1f} s, VRAM {vram_used_gb():.1f} Go")
    for lang, text in TTS_TEXTS:
        firsts, rtfs = [], []
        for _ in range(runs):
            chunks, done = [], threading.Event()
            first = []
            start = time.perf_counter()

            def on_chunk(pcm, chunks=chunks, first=first, start=start):
                if not first:
                    first.append((time.perf_counter() - start) * 1000)
                chunks.append(pcm)

            tts.submit(TTSJob(text, lang, on_chunk=on_chunk, on_done=done.set))
            done.wait()
            total = time.perf_counter() - start
            seconds = sum(len(c) for c in chunks) / 2 / 24000
            firsts.append(first[0])
            rtfs.append(total / max(seconds, 1e-6))
        print(f"  TTS [{lang}] {text[:40]!r}: premier son {med(firsts)}, RTF {statistics.median(rtfs):.2f}")
    return tts


def bench_stt(cfg: dict, runs: int):
    from server.stt import ParakeetSTT

    t0 = time.perf_counter()
    stt = ParakeetSTT(cfg["stt"])
    stt.warmup()
    print(f"STT chargé + préchauffé en {time.perf_counter() - t0:.1f} s, VRAM {vram_used_gb():.1f} Go")
    rng = np.random.default_rng(0)
    for seconds in (2, 5, 10):
        audio = (rng.standard_normal(16000 * seconds) * 0.05).astype(np.float32)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            stt.transcribe(audio)
            times.append((time.perf_counter() - t) * 1000)
        print(f"  STT {seconds:>2} s d'audio : {med(times)}")
    return stt


async def bench_llm(cfg: dict, runs: int) -> None:
    from server.llm import LlamaServer, LLMClient

    t0 = time.perf_counter()
    LlamaServer(cfg["llm"]).start()
    print(f"LLM prêt en {time.perf_counter() - t0:.1f} s, VRAM {vram_used_gb():.1f} Go")
    client = LLMClient(cfg["llm"])
    system = {"role": "system", "content": load_subjects()["anglais"].system_prompt("", "Test")}
    t = time.perf_counter()
    await client.complete([system, {"role": "user", "content": "Hi"}], max_tokens=1)
    print(f"  Prefill à froid du system prompt : {(time.perf_counter() - t) * 1000:.0f} ms")
    ttfts, speeds = [], []
    for i in range(runs):
        t = time.perf_counter()
        first, n = None, 0
        async for _ in client.stream_chat([system, {"role": "user", "content": f"Yesterday I goed to the park ({i})."}]):
            first = first or time.perf_counter()
            n += 1
        ttfts.append((first - t) * 1000)
        speeds.append(n / max(time.perf_counter() - first, 1e-6))
    print(f"  LLM premier token (préfixe en cache) : {med(ttfts)}, ~{statistics.median(speeds):.0f} morceaux/s")
    await client.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    runs = ap.parse_args().runs
    cfg = load_config()
    print(f"VRAM au départ : {vram_used_gb():.1f} Go")
    asyncio.run(bench_llm(cfg, runs))
    stt = bench_stt(cfg, runs)
    tts = bench_tts(cfg, runs)
    print(f"\nVRAM totale, tout chargé : {vram_used_gb():.1f} Go (objectif < 22 Go)")
    del stt, tts


if __name__ == "__main__":
    main()
