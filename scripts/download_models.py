"""Télécharge les modèles (cache Hugging Face) et les binaires llama-server.

Usage : python scripts/download_models.py [core|bench|all]
  core  : ce que l'appli charge en VRAM + VoiceDesign (création de la voix du prof)
  bench : variantes à comparer au banc d'essai (étape 1)
"""

import http.client
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = ROOT / "tools" / "llama.cpp"
LLAMA_BUILD = "b11195"
LLAMA_ZIPS = [
    f"llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
]

# (repo, fichier unique) ou (repo, None) pour tout le dépôt
CORE = [
    ("google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "gemma-4-26B_q4_0-it.gguf"),
    ("unsloth/gemma-4-26B-A4B-it-qat-GGUF", "mtp-gemma-4-26B-A4B-it.gguf"),
    ("istupakov/parakeet-tdt-0.6b-v3-onnx", None),
    ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"),
    ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", None),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", None),
]
BENCH = [
    ("unsloth/gemma-4-26B-A4B-it-qat-GGUF", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"),
    ("google/gemma-4-12B-it-qat-q4_0-gguf", "gemma-4-12b-it-qat-q4_0.gguf"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None),
]


def fetch(repo: str, filename: str | None) -> None:
    print(f"-> {repo} {filename or '(dépôt complet)'}", flush=True)
    if filename:
        path = hf_hub_download(repo, filename)
    else:
        path = snapshot_download(repo, allow_patterns=None)
    print(f"   {path}", flush=True)


def fetch_llama_server() -> None:
    if (LLAMA_DIR / "llama-server.exe").exists():
        print(f"-> llama-server déjà présent dans {LLAMA_DIR}")
        return
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    for name in LLAMA_ZIPS:
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{name}"
        print(f"-> {url}", flush=True)
        zip_path = LLAMA_DIR / name
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp, open(zip_path, "wb") as out:
                    shutil.copyfileobj(resp, out, length=1 << 20)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(LLAMA_DIR)
                break
            except (OSError, http.client.HTTPException, zipfile.BadZipFile) as exc:
                print(f"   tentative {attempt} échouée : {exc}", flush=True)
                if attempt == 5:
                    raise
        zip_path.unlink()
    print(f"   {LLAMA_DIR}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "core"
    items = {"core": CORE, "bench": BENCH, "all": CORE + BENCH}[target]
    if target in ("core", "all"):
        fetch_llama_server()
    for repo, filename in items:
        fetch(repo, filename)
    print("Terminé.")


if __name__ == "__main__":
    main()
