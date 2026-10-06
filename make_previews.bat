@echo off
rem Drag a model file (or a folder of models) onto this file to render TurboSquid previews.
if "%~1"=="" (
    echo Usage: drag a .glb/.fbx/.obj file or a folder onto make_previews.bat
    pause
    exit /b 1
)
python "%~dp0make_previews.py" %*
pause
