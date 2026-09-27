"""Qwen3-TTS 0.6B Base (faster-qwen3-tts, CUDA graphs) avec la voix clonée du prof.

Un seul thread possède le modèle (CUDA graphs non réentrants) et traite les segments
dans l'ordre ; chaque segment produit des morceaux PCM16 24 kHz au fil de l'eau.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS

from server import health

log = logging.getLogger("profs.tts")

LANGUAGES = {"en": "English", "fr": "French"}


@dataclass
class TTSJob:
    text: str
    lang: str
    on_chunk: Callable[[bytes], None]  # appelé depuis le thread TTS
    on_done: Callable[[], None]
    voice: str | None = None  # id de matière ; None = première voix chargée
    cancel: threading.Event = field(default_factory=threading.Event)


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class TeacherTTS:
    """Un seul modèle résident ; une empreinte vocale (x-vector) par prof, calculée au démarrage."""

    def __init__(self, cfg: dict, voices: dict[str, Path]):
        self.cfg = cfg
        self.chunk_size = cfg.get("chunk_size", 4)
        self.model = FasterQwen3TTS.from_pretrained(cfg["repo"])
        self.voices = {}
        for name, ref in voices.items():
            if not ref.exists():
                log.warning("Voix %r absente (%s) : lancer scripts/design_voice.py %s", name, ref, name)
                continue
            items = self.model.model.create_voice_clone_prompt(ref_audio=str(ref), ref_text="",
                                                               x_vector_only_mode=True)
            # x-vector seul : pas de fuite de phonèmes de la référence, changement de langue propre
            self.voices[name] = dict(ref_code=[None], ref_spk_embedding=[items[0].ref_spk_embedding],
                                     x_vector_only_mode=[True], icl_mode=[False])
        if not self.voices:
            raise FileNotFoundError("Aucune voix de prof : lancer scripts/design_voice.py")
        self.default_voice = next(iter(self.voices))
        self.jobs: queue.Queue[TTSJob] = queue.Queue()
        threading.Thread(target=self._worker, name="tts", daemon=True).start()

    def stream(self, text: str, lang: str, voice: str | None = None):
        """Générateur synchrone de morceaux float32 (à appeler depuis le thread TTS)."""
        return self.model.generate_voice_clone_streaming(
            text=text, language=LANGUAGES.get(lang, "English"),
            voice_clone_prompt=self.voices.get(voice) or self.voices[self.default_voice],
            chunk_size=self.chunk_size, temperature=self.cfg.get("temperature", 0.8),
        )

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self.model.warmup()
        done = threading.Event()
        for text, lang in (("Hello, how are you today?", "en"), ("Bonjour, comment ça va ?", "fr")):
            done.clear()
            self.submit(TTSJob(text, lang, on_chunk=lambda _: None, on_done=done.set))
            done.wait()
        log.info("TTS préchauffé en %.1f s", time.perf_counter() - t0)

    def submit(self, job: TTSJob) -> None:
        self.jobs.put(job)

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job.cancel.is_set():
                    continue
                gen = self.stream(job.text, job.lang, job.voice)
                try:
                    with torch.inference_mode():
                        for audio, _sr, _timing in gen:
                            if job.cancel.is_set():
                                break
                            if len(audio):
                                job.on_chunk(to_pcm16(audio))
                finally:
                    gen.close()
            except Exception as exc:
                log.exception("Échec TTS sur %r", job.text)
                if health.is_gpu_fatal(exc):
                    health.fatal("TTS")
            finally:
                job.on_done()
