@echo off
REM ==========================================================================
REM  Create the "alpr" conda environment (Python 3.13) from the offline wheels
REM  in ..\packages\python.  Run from an Anaconda Prompt.
REM ==========================================================================
setlocal
cd /d "%~dp0"
call conda create -n alpr python=3.13 pip -y || goto :err
call conda activate alpr || goto :err
python -m pip install --no-index --find-links ..\packages\python -r requirements.txt || (
  echo Offline wheels incomplete - downloading from the internet instead
  python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu || goto :err
)
python -c "import torch, ultralytics, onnxruntime, cv2; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
echo Environment ready:  conda activate alpr
exit /b 0
:err
echo *** setup failed ***
exit /b 1
