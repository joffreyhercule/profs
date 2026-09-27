"""Parakeet TDT 0.6B v3 (ONNX, GPU). Assez rapide pour re-transcrire tout l'énoncé
en cours toutes les ~300 ms : cela tient lieu de streaming, avec un seul modèle."""

import logging
import time

import numpy as np
import onnx_asr
import onnxruntime as ort
import torch  # noqa: F401  charge les DLL CUDA 13 (cuBLAS, cuDNN) dont le provider CUDA d'onnxruntime a besoin
from huggingface_hub import snapshot_download

from server.config import resolve

log = logging.getLogger("profs.stt")

SAMPLE_RATE = 16000
# Allocation au plus juste : ~0,5 Go de VRAM économisés par rapport aux réglages par défaut
CUDA_OPTIONS = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "0"}
# La réserve mémoire d'onnxruntime garde une zone par taille d'entrée. En mains libres, l'énoncé en
# cours est retranscrit toutes les ~300 ms en s'allongeant : un énoncé de 45 s ajoutait +21 Go de
# VRAM (mesuré), qui débordaient en RAM système et ralentissaient tout le GPU (LLM à 6 tokens/s).
# On rend donc au GPU, après chaque appel, les zones inutilisées : ~+3 ms par transcription.
SHRINK_ARENA = ort.RunOptions()
SHRINK_ARENA.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")


class ParakeetSTT:
    def __init__(self, cfg: dict):
        path = resolve(cfg["path"]) if cfg.get("path") else snapshot_download(cfg["repo"])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # erreurs seulement
        opts.intra_op_num_threads = 2
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # pas d'attente active
        self.model = onnx_asr.load_model(
            cfg["model_name"], path,
            quantization=cfg.get("quantization"),
            sess_options=opts,
            providers=[("CUDAExecutionProvider", CUDA_OPTIONS), "CPUExecutionProvider"],
        )
        self.providers = self.model.asr._encoder.get_providers()
        if "CUDAExecutionProvider" not in self.providers:
            log.warning("STT sur CPU (%s) : la latence sera mauvaise", self.providers)
        else:
            # les deux sessions dont la taille d'entrée suit la durée de l'audio (le décodeur, lui,
            # avance trame par trame avec des tailles fixes)
            preprocessor = getattr(self.model.asr._preprocessor, "_preprocessor", None)
            for session in (self.model.asr._encoder, preprocessor):
                if isinstance(session, ort.InferenceSession):
                    self._shrink_after_run(session)

    @staticmethod
    def _shrink_after_run(session: ort.InferenceSession) -> None:
        """onnx_asr appelle session.run sans options : on les ajoute."""
        run = session.run
        session.run = lambda output_names, input_feed, run_options=None: run(output_names, input_feed,
                                                                             SHRINK_ARENA)

    def transcribe(self, audio: np.ndarray) -> str:
        """audio : float32 mono 16 kHz."""
        if len(audio) < SAMPLE_RATE // 10:
            return ""
        return self.model.recognize(audio.astype(np.float32, copy=False), sample_rate=SAMPLE_RATE).strip()

    def warmup(self) -> None:
        t0 = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 3) * 0.01).astype(np.float32)
        for _ in range(3):
            self.transcribe(noise)
        log.info("STT préchauffé en %.1f s", time.perf_counter() - t0)
