"""Convertit l'encodeur Parakeet ONNX en fp16 (tensor cores) : plus rapide et deux fois plus léger.

Écrit data/models/parakeet-tdt-0.6b-v3-fp16/ au format attendu par onnx-asr
(quantization="fp16"). Le décodeur, minuscule, reste en fp32.
Usage : python scripts/convert_stt_fp16.py
"""

import shutil
import sys
from pathlib import Path

import onnx
from huggingface_hub import snapshot_download
from onnxruntime.transformers.float16 import convert_float_to_float16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import ROOT, load_config  # noqa: E402


def main() -> None:
    src = Path(snapshot_download(load_config()["stt"]["repo"]))
    dst = ROOT / "data" / "models" / "parakeet-tdt-0.6b-v3-fp16"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "vocab.txt", "nemo128.onnx"):
        shutil.copy(src / name, dst / name)
    shutil.copy(src / "decoder_joint-model.onnx", dst / "decoder_joint-model.fp16.onnx")

    model = onnx.load(str(src / "encoder-model.onnx"), load_external_data=True)
    model16 = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    onnx.save(model16, str(dst / "encoder-model.fp16.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="encoder-model.fp16.onnx.data")
    size = sum(f.stat().st_size for f in dst.glob("encoder-model.fp16.onnx*")) / 1e9
    print(f"Encodeur fp16 : {dst} ({size:.2f} Go)")


if __name__ == "__main__":
    main()
