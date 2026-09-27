@echo off
chcp 65001 >nul
REM Lance le prof : vérifie la VRAM libre, puis démarre le serveur, qui lance llama-server,
REM charge et préchauffe tous les modèles, puis ouvre le navigateur.
REM Si Windows réinitialise le GPU (TDR), le serveur sort avec le code 3 et on le relance.
REM Arrêt : Ctrl+C. Fermer la fenêtre laisserait llama-server tourner en arrière-plan.
title Profs
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :not_installed
if not exist "tools\llama.cpp\llama-server.exe" goto :not_installed
where nvidia-smi >nul 2>nul || goto :no_gpu

REM Tout est déjà en cache : aucun accès réseau au démarrage.
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:loop
REM Un llama-server resté d'une séance précédente est réutilisé : sa VRAM est déjà prise.
tasklist /fi "imagename eq llama-server.exe" | find /i "llama-server.exe" >nul && goto :serve
set FREE=0
for /f %%v in ('nvidia-smi --query-gpu^=memory.free --format^=csv^,noheader^,nounits') do set FREE=%%v
if %FREE% GEQ 21000 goto :serve
echo Seulement %FREE% Mo de VRAM libres : il en faut ~21 000. Décharge ComfyUI ou tout autre modèle, puis relance.
goto :end

:serve
.venv\Scripts\python.exe -m server.main
if not "%ERRORLEVEL%"=="3" goto :end
echo Le GPU a été réinitialisé par Windows : redémarrage du prof…
REM La page déjà ouverte se reconnecte toute seule.
set PROFS_NO_BROWSER=1
timeout /t 3 /nobreak >nul
goto :loop

:not_installed
echo L'installation n'est pas faite : lance d'abord install.bat.
goto :end

:no_gpu
echo nvidia-smi est introuvable : il faut un GPU NVIDIA et son pilote.

:end
pause
