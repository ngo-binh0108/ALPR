@echo off
REM ==========================================================================
REM  Open LabelMe on a folder of vehicle photos (review / correct plate labels)
REM    label_tool.bat D:\site_photos
REM  First time: creates the "labelme" conda env from ..\packages\python_labelme
REM ==========================================================================
setlocal
cd /d "%~dp0"
call conda activate labelme 2>nul || (
  echo Creating conda env "labelme" ...
  call conda create -n labelme python=3.13 pip -y || goto :err
  call conda activate labelme || goto :err
  python -m pip install --no-index --find-links ..\packages\python_labelme labelme || python -m pip install labelme || goto :err
)
labelme %*
exit /b 0
:err
echo *** could not start labelme ***
exit /b 1
