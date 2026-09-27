# profs : des profs vocaux, 100 % local

Tu choisis ton profil et ta matière, puis tu parles : le prof te répond à voix haute et te corrige. Chaque élève a sa propre mémoire, séparée par matière : erreurs récurrentes, notions à revoir, niveau et bilans de séance.

- **Anglais avec Claire** : tu parles en anglais, ou en français quand tu bloques. Elle corrige grammaire, conjugaison et vocabulaire.
- **Botanique avec Basile** : cours oral en français, une notion à la fois, avec des questions. Il corrige tes idées fausses et revient sur ce qu'il faut revoir.

Tout tourne sur la machine, sur un GPU de 24 Go. Les modèles restent chargés en VRAM et ne sont jamais rechargés ; ajouter une matière ne coûte pas de VRAM.

| Rôle | Modèle | Runtime |
|---|---|---|
| Fin de tour | Silero VAD + smart-turn v3.2 | CPU |
| Transcription | Parakeet TDT 0.6B v3 | ONNX Runtime CUDA |
| Profs | Gemma 4 26B-A4B QAT Q4_0 + brouillon MTP | llama-server (llama.cpp CUDA 13) |
| Voix | Qwen3-TTS 0.6B, une voix par prof créée par Qwen3-TTS VoiceDesign | faster-qwen3-tts (CUDA graphs) |

## Installation

Il faut Windows 11, un GPU NVIDIA avec ~23 Go de VRAM libres (testé sur une RTX 5090 Laptop de 24 Go) et un pilote récent compatible CUDA 13, git, et ~30 Go de disque : 24 Go de modèles, 5,5 Go d'environnement Python, 0,7 Go pour llama.cpp.

```powershell
git clone https://github.com/joffreyhercule/profs
cd profs
.\install.bat
```

`install.bat` installe uv (qui installe lui-même Python 3.12 s'il manque), crée l'environnement `.venv`, puis télécharge llama-server (llama.cpp CUDA) et les modèles depuis Hugging Face. Aucun compte ni jeton n'est nécessaire : tous les modèles sont publics. Tu peux le relancer sans risque : ce qui est déjà téléchargé est gardé.

Les voix de Claire et de Basile sont fournies dans `data\voices\`. `design_voice.py` ne sert qu'à en créer une nouvelle ou à en recréer une, serveur arrêté (VoiceDesign 1.7B prend ~5 Go de VRAM) :

```powershell
.\.venv\Scripts\python.exe scripts\design_voice.py botanique "autre description"
```

Écoute les essais dans `data\voices\samples\`. Une matière dont la voix n'existe pas encore n'est pas proposée.

## Lancer une séance

Double-clique sur `run.bat`, ou lance `.\run.bat` dans un terminal. Pour l'avoir sur le bureau : clic droit sur `run.bat` > Afficher d'autres options > Envoyer vers > Bureau (créer un raccourci).

Le script vérifie qu'il reste ~23 Go de VRAM libres (décharge ComfyUI ou tout autre modèle avant), charge tout, puis ouvre http://127.0.0.1:8765. Arrête le prof avec **Ctrl+C** dans sa fenêtre : si tu la fermes avec la croix, llama-server continue de tourner en arrière-plan et garde sa VRAM (`Get-Process llama-server | Stop-Process` pour l'arrêter). Choisis ou crée ton profil, choisis ta matière, puis « Commencer la séance ». Le navigateur retient ton dernier choix.

- **Mains libres** : parle, le prof répond dès que tu as fini ta phrase. Parle par-dessus lui pour l'interrompre.
- **Espace maintenue** (ou le bouton micro en mode « Appuyer pour parler ») : la fin de tour est immédiate au relâchement, c'est la latence la plus basse.
- **Échap** coupe le prof.
- **Terminer la séance** : le prof rédige un bilan qui alimente la séance suivante.

La mémoire est dans `data\profs.db` (SQLite). Pour repartir de zéro, supprime ce fichier serveur arrêté : il est recréé vide au lancement.

## Ajouter une matière

Copie `subjects\botanique.yaml` sous un nouveau nom et adapte-le : identifiant, nom du prof, description et langue de sa voix, types d'erreurs suivis, consignes pédagogiques (`prompt`), salutation, bilan de fin de séance. Garde le format de sortie `<say>` / `<fix>` décrit dans le prompt. Puis crée sa voix avec `design_voice.py <identifiant>` et relance le serveur.

## Bancs d'essai

```powershell
.\.venv\Scripts\python.exe scripts\bench_components.py          # latence de chaque étage + VRAM totale
.\.venv\Scripts\python.exe scripts\bench_stt_verbatim.py record  # enregistre tes phrases fautives
.\.venv\Scripts\python.exe scripts\bench_stt_verbatim.py eval    # la transcription garde-t-elle tes fautes ?
.\.venv\Scripts\python.exe scripts\bench_llm_corrections.py      # détection des fautes, surcorrection, format
.\.venv\Scripts\python.exe scripts\bench_e2e.py --mode ptt       # fin de parole -> premier son (serveur lancé, voir ci-dessous)
.\.venv\Scripts\python.exe -m pytest                             # parseur, mémoire, pipeline (sans GPU)
```

Les phrases de test sont dans `tests\data\sentences.yaml`. `bench_stt_verbatim.py synth` les fait lire par la voix TTS si tu ne veux pas les enregistrer.

Pour `bench_e2e.py`, lance le serveur sur une base à part, sinon les séances de test alimentent ta mémoire d'élève :

```powershell
$env:PROFS_DB = "data\bench.db"; $env:PROFS_NO_BROWSER = "1"; .\.venv\Scripts\python.exe -m server.main
```

## Performances mesurées (RTX 5090 Laptop, 26/09/2026)

| | Push-to-talk | Mains libres |
|---|---|---|
| Fin de parole → premier son, p50 | 317 ms | 362 ms |
| p95 | 385 ms | ~1,4 s (quand smart-turn juge la phrase inachevée, on attend 1,5 s de silence) |
| Trous dans l'audio | aucun | aucun |

VRAM totale : 22,8 Go (contexte du LLM de 12 288 tokens). Détection des fautes : 87/90 ; aucune surcorrection sur 45 phrases justes.

Ce qui a compté, dans l'ordre :

- Limiter les fils CPU de torch et d'onnxruntime (sans attente active) : leurs pools tournaient à vide et affamaient les fils qui pilotent le GPU (−230 ms en mains libres).
- Mettre le LLM en pause après le premier segment, le temps que la voix sorte son premier son, puis reprendre (−110 ms).
- Construire les prompts nous-mêmes (`ChatFormat`) : le modèle de chat de Gemma 4 réécrit les réponses passées, ce qui cassait le cache de préfixe à chaque tour.
- Lancer la transcription dès 64 ms de silence, et le tour spéculatif dès 160 ms ; le prof ne parle qu'après 300 ms de silence.

## Réglages

Tout est dans `config.yaml` : modèles, seuils de détection de fin de tour (`vad`), taille des morceaux audio (`tts.chunk_size`), budget d'historique (`memory`).
