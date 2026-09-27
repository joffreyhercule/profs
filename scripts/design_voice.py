"""Crée la voix d'un prof une fois pour toutes avec Qwen3-TTS VoiceDesign (1.7B).

La description, la langue et le texte de référence viennent de subjects/<matière>.yaml.
Le modèle Base (0.6B) résident clone ensuite cette voix, quelle que soit la langue.
Le VoiceDesign n'est pas gardé en VRAM : lancer ce script serveur arrêté.

Usage : python scripts/design_voice.py <matière> ["autre description de voix"]
Écrit le fichier `voice` de la matière, et un essai dans data/voices/samples/.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

from server.config import ROOT  # noqa: E402
from server.subjects import load_subjects  # noqa: E402


def as_wav(audio) -> np.ndarray:
    return np.asarray(audio[0] if isinstance(audio, list) else audio, dtype=np.float32).flatten()


def main() -> None:
    subjects = load_subjects()
    if len(sys.argv) < 2 or sys.argv[1] not in subjects:
        sys.exit(f"Usage : design_voice.py <{'|'.join(subjects)}> [description]")
    subject = subjects[sys.argv[1]]
    description = sys.argv[2] if len(sys.argv) > 2 else subject.voice_description
    samples = ROOT / "data" / "voices" / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    design = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    audio, sr = design.generate_voice_design(text=subject.voice_sample, instruct=description,
                                             language=subject.voice_language)
    audio = as_wav(audio)
    subject.voice.parent.mkdir(parents=True, exist_ok=True)
    sf.write(subject.voice, audio, sr)
    print(f"Voix de {subject.teacher} : {subject.voice} ({len(audio) / sr:.1f} s)")
    del design
    torch.cuda.empty_cache()

    # Essai de la voix clonée par le modèle résident, comme en séance
    base = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    wavs, sr = base.generate_voice_clone(text=subject.voice_sample, language=subject.voice_language,
                                         ref_audio=str(subject.voice), ref_text="", xvec_only=True)
    trial = samples / f"essai_{subject.id}.wav"
    sf.write(trial, as_wav(wavs), sr)
    print(f"Essai : {trial}")


if __name__ == "__main__":
    main()
