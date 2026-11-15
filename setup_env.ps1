# =============================================================
#  RSNA Pneumonia Detection — 一鍵建立環境（Windows / PowerShell）
#
#  用法（在專案根目錄開 PowerShell）：
#      powershell -ExecutionPolicy Bypass -File setup_env.ps1
#
#  想重建環境：先執行  conda env remove -n rsna
# =============================================================

$ErrorActionPreference = "Stop"
$EnvName = "rsna"
$CudaIndex = "https://download.pytorch.org/whl/cu126"   # CUDA 12.6 wheels

Write-Host "=== [1/5] 檢查 conda ===" -ForegroundColor Cyan
conda --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "找不到 conda。請先安裝 Miniforge / Miniconda：" -ForegroundColor Red
    Write-Host "  https://github.com/conda-forge/miniforge" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== [2/5] 建立 conda 環境 '$EnvName' (python 3.11) ===" -ForegroundColor Cyan
$existing = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if ($existing) {
    Write-Host "環境 '$EnvName' 已存在，跳過建立。（要重建請先 conda env remove -n $EnvName）" -ForegroundColor Yellow
} else {
    conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) { Write-Host "conda 環境建立失敗" -ForegroundColor Red; exit 1 }
}

# 取得該環境的 python 路徑，之後都用它來跑 pip，避免 activate 的麻煩
$EnvPrefix = (conda run -n $EnvName python -c "import sys; print(sys.prefix)").Trim()
Write-Host "環境路徑: $EnvPrefix" -ForegroundColor Green

Write-Host "`n=== [3/5] 安裝 PyTorch (CUDA 12.6) ===" -ForegroundColor Cyan
Write-Host "這一步檔案較大（約 2.5 GB），請耐心等待..." -ForegroundColor Yellow
conda run -n $EnvName --no-capture-output python -m pip install --upgrade pip
conda run -n $EnvName --no-capture-output python -m pip install torch torchvision --index-url $CudaIndex
if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA 版 PyTorch 安裝失敗，改裝 CPU 版（訓練會很慢，但至少能跑）" -ForegroundColor Yellow
    conda run -n $EnvName --no-capture-output python -m pip install torch torchvision
}

Write-Host "`n=== [4/5] 安裝其餘套件 ===" -ForegroundColor Cyan
conda run -n $EnvName --no-capture-output python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "套件安裝失敗" -ForegroundColor Red; exit 1 }

Write-Host "`n=== [5/5] 驗證環境 ===" -ForegroundColor Cyan
conda run -n $EnvName --no-capture-output python src/check_env.py

Write-Host "`n完成！接下來請執行：" -ForegroundColor Green
Write-Host "    conda activate $EnvName" -ForegroundColor Green
Write-Host "    python src/download_data.py" -ForegroundColor Green
