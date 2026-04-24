@echo off
echo ============================================
echo   SentimentEdge - Retrain and Deploy
echo ============================================
echo.

:: ── Step 1: Train the model ──
echo [1/4] Training RF model...
cd /d D:\sentimentEdge\backend
python train.py
if %errorlevel% neq 0 (
    echo ERROR: Training failed! Check your train.py
    pause
    exit /b 1
)
echo Training complete!
echo.

:: ── Step 2: Copy files to HF folder ──
echo [2/4] Copying model files to HF folder...
copy /Y D:\sentimentEdge\backend\rf_model.joblib D:\sentimentEdge\hf-sentimentedge\backend\
copy /Y D:\sentimentEdge\backend\model_metrics.json D:\sentimentEdge\hf-sentimentedge\backend\
if %errorlevel% neq 0 (
    echo ERROR: Copy failed! Check file paths.
    pause
    exit /b 1
)
echo Files copied!
echo.

:: ── Step 3: Git add and commit ──
echo [3/4] Committing to git...
cd /d D:\sentimentEdge\hf-sentimentedge
git add .
git commit -m "update: retrained RF model and metrics"
echo Committed!
echo.

:: ── Step 4: Push to HuggingFace ──
echo [4/4] Pushing to HuggingFace...
git push
if %errorlevel% neq 0 (
    echo ERROR: Push failed! Check your HF credentials.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Done! HF Space will restart in ~2 mins
echo   https://huggingface.co/spaces/vectorxx/sentiment
echo ============================================
pause