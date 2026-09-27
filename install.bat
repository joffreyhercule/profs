@echo off
chcp 65001 >nul
REM Installe tout ce qu'il faut pour lancer les profs (ensuite : run.bat) :
REM   1. uv, qui installe aussi Python 3.12 s'il manque, puis l'environnement Python (.venv)
REM   2. llama-server (llama.cpp CUDA) dans tools\llama.cpp
REM   3. les modèles (~24 Go, cache Hugging Face) ; pas de jeton : ils sont tous publics
REM Relançable : ce qui est déjà présent n'est pas retéléchargé.
title Profs - installation
cd /d "%~dp0"

where uv >nul 2>nul || goto :uv_local
set "UV=uv"
goto :sync
:uv_local
if not exist "%USERPROFILE%\.local\bin\uv.exe" goto :uv_pip
set UV="%USERPROFILE%\.local\bin\uv.exe"
goto :sync
:uv_pip
python -m uv --version >nul 2>nul || goto :uv_install
set "UV=python -m uv"
goto :sync
:uv_install
echo == Installation de uv
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if not exist "%USERPROFILE%\.local\bin\uv.exe" goto :fail
set UV="%USERPROFILE%\.local\bin\uv.exe"

:sync
echo == Environnement Python (.venv)
%UV% sync --group dev
if errorlevel 1 goto :fail

echo == llama-server et modèles
REM Le protocole Xet de Hugging Face peut se bloquer : on télécharge en HTTP classique.
set HF_HUB_DISABLE_XET=1
.venv\Scripts\python.exe scripts\download_models.py core
if errorlevel 1 goto :fail

echo.
echo Installation terminée : lance run.bat.
pause
exit /b 0

:fail
echo.
echo L'installation a échoué, voir le message ci-dessus. Relance install.bat : ce qui est déjà téléchargé est gardé.
pause
exit /b 1
