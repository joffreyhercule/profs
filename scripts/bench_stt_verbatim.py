"""Banc STT : les fautes de l'élève survivent-elles à la transcription ?

  python scripts/bench_stt_verbatim.py synth           # lit les phrases avec la voix TTS -> data/recordings/synth/
  python scripts/bench_stt_verbatim.py record          # enregistre TA voix au micro   -> data/recordings/me/
  python scripts/bench_stt_verbatim.py eval [dossier]  # transcrit et vérifie (défaut : data/recordings/me)
  options d'eval : --quant int8   (compare la version int8 de Parakeet)
"""

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import ROOT, load_config  # noqa: E402

SENTENCES = yaml.safe_load((ROOT / "tests" / "data" / "sentences.yaml").read_text(encoding="utf-8"))
REC = ROOT / "data" / "recordings"


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w' ]+", " ", s.lower().replace("’", "'")).split())


def load_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return soxr.resample(audio, sr, 16000) if sr != 16000 else audio


def synth() -> None:
    import threading

    from server.subjects import load_subjects
    from server.tts import TeacherTTS, TTSJob

    subject = load_subjects()["anglais"]
    tts = TeacherTTS(load_config()["tts"], {subject.id: subject.voice})
    out = REC / "synth"
    out.mkdir(parents=True, exist_ok=True)
    for item in SENTENCES["wrong"] + SENTENCES["correct"]:
        chunks, done = [], threading.Event()
        lang = "fr" if item["id"] == "c13" else "en"
        tts.submit(TTSJob(item["text"], lang, on_chunk=chunks.append, on_done=done.set))
        done.wait()
        pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
        sf.write(out / f"{item['id']}.wav", pcm, 24000, subtype="PCM_16")
        print(f"{item['id']} : {len(pcm) / 24000:.1f} s")


def record() -> None:
    import sounddevice as sd

    out = REC / "me"
    out.mkdir(parents=True, exist_ok=True)
    print("Pour chaque phrase : Entrée pour démarrer, lis la phrase naturellement, Entrée pour arrêter.")
    print("Tape s puis Entrée pour passer une phrase, q pour quitter.\n")
    for item in SENTENCES["wrong"] + SENTENCES["correct"]:
        print(f"[{item['id']}] {item['text']}")
        if input("  prêt ? ").strip().lower() in ("s", "q"):
            continue
        frames = []
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=lambda d, *_: frames.append(d.copy())):
            input("  enregistrement… Entrée pour arrêter ")
        sf.write(out / f"{item['id']}.wav", np.concatenate(frames)[:, 0], 16000, subtype="PCM_16")


def evaluate(folder: Path, quant: str | None) -> None:
    from server.stt import ParakeetSTT

    cfg = load_config()["stt"]
    cfg["quantization"] = quant
    stt = ParakeetSTT(cfg)
    stt.warmup()
    kept, total, times, over = 0, 0, [], 0
    for item in SENTENCES["wrong"] + SENTENCES["correct"]:
        path = folder / f"{item['id']}.wav"
        if not path.exists():
            continue
        audio = load_16k(path)
        t0 = time.perf_counter()
        text = stt.transcribe(audio)
        times.append((time.perf_counter() - t0) * 1000)
        if "error" in item:
            total += 1
            ok = norm(item["error"]) in norm(text)
            kept += ok
            flag = "gardée " if ok else "EFFACÉE"
            print(f"{flag} {item['id']} attendu « {item['error']} » -> {text!r}")
        else:
            print(f"        {item['id']} {text!r}")
    if total:
        print(f"\nFautes conservées : {kept}/{total} ({100 * kept / total:.0f} %)")
    if times:
        print(f"Temps de transcription : médiane {statistics.median(times):.0f} ms, max {max(times):.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["synth", "record", "eval"])
    ap.add_argument("folder", nargs="?", default=str(REC / "me"))
    ap.add_argument("--quant", default=None)
    args = ap.parse_args()
    if args.action == "synth":
        synth()
    elif args.action == "record":
        record()
    else:
        evaluate(Path(args.folder), args.quant)
