@echo off
REM ==========================================================================
REM  Full few-data training pipeline (run inside:  conda activate alpr)
REM    run_all.bat          -> CPU
REM    run_all.bat 0        -> NVIDIA GPU 0 (much faster)
REM ==========================================================================
setlocal
cd /d "%~dp0"
set DEVICE=cpu
if not "%1"=="" set DEVICE=%1

echo ===== 1/5 prepare a small dataset (800 train / 200 val images) =====
python 01_prepare_data.py ^
  --train ../datasets/raw/ccpd2019/trainn:600 --train ../datasets/raw/CCPD2020/ccpd_green/train:200 ^
  --val   ../datasets/raw/ccpd2019/valn:150   --val   ../datasets/raw/CCPD2020/ccpd_green/val:50 || goto :err

echo ===== 2/5 recogniser: synthetic pre-training + real fine-tuning =====
python 03_train_recognizer.py --synth 40000 --epochs-a 3 --epochs-b 25 --steps-b 60 || goto :err

echo ===== 3/5 detector: YOLO pose (plate + 4 corners) =====
python 02_train_detector.py --epochs 30 --imgsz 512 --device %DEVICE% || goto :err

echo ===== 4/5 export ONNX + model contract for the C++ app =====
python 04_export_onnx.py --imgsz 512 || goto :err

echo ===== 5/5 end-to-end evaluation on unseen CCPD images =====
python 05_evaluate.py --images ../datasets/raw/CCPD2020/ccpd_green/test --limit 300
python 05_evaluate.py --images ../datasets/raw/ccpd2019/valn --limit 300

echo Done. Models are in ..\models  (copied next to ALPRStudio.exe by the C++ build)
exit /b 0
:err
echo *** pipeline failed ***
exit /b 1
