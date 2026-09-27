# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Profs vocaux 100 % locaux (page web sur localhost) : l'élève choisit son profil et une matière (anglais avec « Claire », botanique avec « Basile »…), parle, et le prof répond à voix haute en le corrigeant, avec une mémoire persistante par élève et par matière. **La contrainte n°1 est la latence** fin de parole → premier son. Tous les modèles restent résidents en VRAM (24 Go, RTX 5090 Laptop, Windows 11 natif).

Référence mesurée (voir README) : p50 ~317 ms en push-to-talk, ~362 ms en mains libres, 21,9 Go de VRAM. Tout changement touchant le chemin critique se mesure avant/après avec `scripts/bench_e2e.py`.

## Commandes (PowerShell, Windows)

```powershell
.\install.bat                                                   # uv + .venv (--group dev) + llama-server + modèles ; relançable
python -m uv sync --group dev                                   # environnement (.venv) seul
.\.venv\Scripts\python.exe -m pytest                            # tests, sans GPU (faux moteurs)
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py::test_barge_in_keeps_only_what_was_heard
.\run.bat                                                       # lance le prof (vérifie la VRAM libre, ouvre le navigateur)
```

Un clone doit marcher avec `install.bat` puis `run.bat`, sans autre étape : les voix des profs (`data/voices/*.wav`) sont versionnées, les modèles HF sont publics (pas de jeton), et tout ce qu'on ajoute au chemin de chargement doit être téléchargé par `download_models.py core`. Les `.bat` sont en UTF-8 avec `chcp 65001` et en CRLF (`.gitattributes`) ; pas de parenthèses dans un `echo` placé dans un bloc `( … )`.

Serveur pour les bancs (base séparée, sans navigateur, cache HF hors ligne) :

```powershell
$env:HF_HUB_OFFLINE="1"; $env:PROFS_DB="data\bench.db"; $env:PROFS_NO_BROWSER="1"; .\.venv\Scripts\python.exe -m server.main
.\.venv\Scripts\python.exe scripts\bench_e2e.py data\recordings\synth --mode ptt        # ou --mode handsfree
```

Autres scripts : `bench_components.py` (latence + VRAM par étage), `bench_llm_corrections.py` (détection/surcorrection/format sur `tests/data/sentences.yaml`, matière anglais), `bench_stt_verbatim.py synth|record|eval` (le STT garde-t-il les fautes ?), `download_models.py core|bench` (avec `HF_HUB_DISABLE_XET=1` : Xet se bloque sur cette machine), `design_voice.py <matière>` (crée la voix du prof, serveur arrêté car VoiceDesign 1.7B prend ~5 Go), `convert_stt_fp16.py`. `bench_e2e.py --subject <matière>` crée/réutilise un profil « Banc d'essai ». Pas de linter configuré.

Contraintes d'exécution :
- Le serveur complet occupe ~22 Go de VRAM : on ne peut pas lancer un banc GPU (`bench_components`, `bench_stt_verbatim`) pendant qu'il tourne. ComfyUI tourne souvent sur cette machine : ne jamais le tuer, demander à l'utilisateur de le décharger.
- `llama-server` (lancé en sous-processus) survit à la mort du processus Python qui l'a lancé : l'arrêter (`Get-Process llama-server | Stop-Process`) avant de relancer, sinon il est réutilisé tel quel (et ne sera plus arrêté à la sortie).

## Architecture

Deux processus : le serveur Python (`server/main.py`, FastAPI) et `llama-server` (llama.cpp CUDA, binaires dans `tools/llama.cpp/`) qu'il lance. Le navigateur envoie des blocs PCM16 16 kHz de 512 échantillons (32 ms) par WebSocket `/ws?user=<id>&subject=<matière>` et reçoit l'audio du prof en binaire : en-tête `struct <II` (tour, segment) + PCM16 24 kHz, plus des événements JSON. La page connecte le WebSocket dès que profil et matière sont choisis (préchauffage du cache LLM) ; la séance en base n'est créée qu'au message `start`, et supprimée à la fin si l'élève n'a rien dit.

### Matières et profils

- Une matière = un fichier `subjects/<id>.yaml` (chargé par `server/subjects.py`) : nom du prof, voix (fichier + description VoiceDesign), langue par défaut, `fix_types`, consignes (`prompt`, `first_session`, `greeting`, `summary`). Il n'y a pas de prompt dans le code. Le format de sortie `<say>/<fix>` est commun à toutes les matières : c'est ce que le parseur et la mémoire exploitent.
- Toutes les matières partagent le même LLM et le même modèle TTS : une voix de plus n'est qu'un x-vector (`TeacherTTS.voices`, `TTSJob.voice`). Une matière dont le fichier voix manque est ignorée au démarrage.
- La mémoire est indexée par (élève, matière) : `error_stats`, `vocabulary`, `learner_profile` ont `(user_id, subject, …)` comme clé. `MemoryDB` migre une base v0 (sans profils) via `PRAGMA user_version`, en rattachant tout à un profil « Mon profil » / anglais, et garde une copie `*.v0-backup.db`.

Chemin critique, orchestré par `Session` dans `server/pipeline.py` (une instance par connexion) :

1. **Détection de tour** (`server/vad.py`, CPU) : Silero VAD par bloc de 32 ms, puis smart-turn v3.2 sur l'audio seul.
2. **Transcription anticipée** : dès `eager_stt_frames` de silence, Parakeet (`server/stt.py`, ONNX CUDA) transcrit l'énoncé ; le résultat est réutilisé par le tour s'il n'y a pas eu de parole depuis.
3. **Tour spéculatif** : dès `min_silence_ms`, un `Turn` *retenu* (`held`) génère tout (LLM + TTS) mais met ses messages en réserve (`outbox`). Il est libéré quand smart-turn dit « fini » ET que le silence atteint `min_release_ms`, ou au bout de `max_silence_ms` ; il est annulé si l'élève reprend. En push-to-talk, le relâchement de la touche lance directement un tour libéré.
4. **LLM** : `LLMClient.stream_prompt` sur `/completion`. **Pause après le premier segment** : on ferme le flux (llama-server arrête de générer), on attend le premier morceau audio (≤ `pause_max_ms`), puis on reprend avec `prompt + turn.raw`. Sinon LLM et TTS se partagent le GPU et le premier son arrive ~110 ms plus tard.
5. **Découpe** (`server/output_parser.py`) : le LLM répond `<say lang="en|fr">…</say>` puis `<fix>[JSON]</fix>`. Le parseur incrémental émet des `Segment` (proposition ou changement de langue via `<en>`/`<fr>`, sinon `guess_lang`) pendant le flux, et relit le JSON de façon tolérante (guillemets non échappés, `]` manquant).
6. **TTS** (`server/tts.py`) : Qwen3-TTS 0.6B via faster-qwen3-tts, **un seul thread propriétaire** (graphes CUDA non réentrants), voix clonée en mode x-vector seul (pas de fuite de phonèmes entre langues), langue passée par segment.
7. **Interruption (barge-in)** : de la parole pendant que le prof parle (seuil relevé contre l'écho) → `flush` côté navigateur, annulation LLM/TTS. Le navigateur accuse réception de chaque segment joué (`seg_start`) et de la fin de lecture (`played`) : l'historique ne garde que ce qui a été entendu.

### Cache de préfixe du LLM (à ne pas casser)

- `ChatFormat` (`server/llm.py`) extrait une fois, via `/apply-template` et des marqueurs, les morceaux du modèle de chat, puis `Session.prompt()` assemble le prompt en **rejouant à l'identique** le texte déjà en cache (prompt précédent + sortie brute). Passer par l'API chat casse le cache : le modèle de chat de Gemma 4 retire des tours passés le canal de réflexion qu'il ajoute à la génération.
- L'historique stocke la **sortie brute du LLM, balises comprises** (`turn.raw`). Avec des réponses passées en texte nu, Gemma abandonne le format `<say>/<fix>` (mesuré : 0/5).
- Le bloc mémoire (`server/memory/profile.py`) et le prénom de l'élève sont construits une fois à la connexion puis figés dans le system prompt (`Subject.system_prompt`). `trim_history` retire d'un coup la moitié la plus ancienne et réchauffe le cache pendant un temps mort.

### Mémoire

SQLite (`server/memory/db.py`, `data/profs.db`) : séances, tours, erreurs, `error_stats` par `rule_key`, vocabulaire, profil, métriques de latence. Chaque `<fix>` alimente les erreurs de façon déterministe. En fin de séance, le LLM (API chat, hors chemin critique) rédige un résumé et estime le niveau CECRL.

### Robustesse

Un reset du GPU par Windows (TDR) invalide tous les contextes CUDA : `server/health.py` sort avec le code 3 et `run.bat` relance tout ; la page se reconnecte seule.

## Invariants et pièges

- `torch.set_num_threads(1)` et `session.intra_op.allow_spinning=0` sur les sessions onnxruntime : sinon les pools de fils CPU tournent à vide et affament les fils qui pilotent le GPU (+230 ms mesurés).
- Importer torch avant de créer une session ORT CUDA : torch cu130 fournit les DLL cuBLAS que `onnxruntime-gpu[cuda]` n'installe pas.
- `transformers==5.15.1` est épinglé (qwen-tts-hf lit `rope_theta`, retiré ensuite) ; `onnxruntime` CPU est exclu par `override-dependencies`, car il écraserait `onnxruntime-gpu`.
- Chronométrer avec `time.perf_counter()` : `time.monotonic()` n'a qu'une résolution de ~15,6 ms sous Windows.
- Pistes déjà essayées et écartées, mesures à l'appui : priorité GPU WDDM, maintien au chaud du GPU, encodeur Parakeet fp16 (2× plus lent), Whisper (corrige les fautes de l'élève).
- Les tests (`tests/test_pipeline.py`) remplacent VAD, STT, LLM, TTS et smart-turn par des faux ; le `FakeLLM` reprend la réponse là où le prompt s'arrête, comme llama-server lors de la reprise après pause.
