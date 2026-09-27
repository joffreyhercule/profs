"""Détection de parole (Silero) et de fin de tour sémantique (smart-turn v3), sur CPU."""

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from silero_vad import load_silero_vad
from transformers import WhisperFeatureExtractor

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms : taille de bloc imposée par Silero à 16 kHz


class SileroVAD:
    """Probabilité de parole par bloc de 32 ms. Une instance par connexion (état interne)."""

    def __init__(self):
        # Un seul fil CPU (recommandation silero) : sinon, appelé toutes les 32 ms, le pool OpenMP de
        # torch tourne à vide sur tous les cœurs et affame les fils qui pilotent le GPU (LLM, TTS).
        torch.set_num_threads(1)
        self.model = load_silero_vad()

    def __call__(self, frame: np.ndarray) -> float:
        with torch.inference_mode():
            return self.model(torch.from_numpy(frame), SAMPLE_RATE).item()

    def reset(self) -> None:
        self.model.reset_states()


class SmartTurn:
    """Probabilité que l'élève ait fini sa phrase, d'après les 8 dernières secondes d'audio."""

    SECONDS = 8

    def __init__(self, cfg: dict):
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # pas d'attente active
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = hf_hub_download(cfg["repo"], cfg["file"])
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=self.SECONDS)

    def __call__(self, audio: np.ndarray) -> float:
        audio = audio[-self.SECONDS * SAMPLE_RATE:]
        inputs = self.features(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
            max_length=self.SECONDS * SAMPLE_RATE, truncation=True, do_normalize=True,
        )
        features = inputs.input_features.squeeze(0).astype(np.float32)[None]
        return float(self.session.run(None, {"input_features": features})[0][0].item())
