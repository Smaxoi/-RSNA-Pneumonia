"""
專案的所有路徑與超參數集中在這裡管理。
其他腳本一律 `from config import ...`，不要在程式裡散落硬編碼路徑。
"""

import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
    """
    Windows 預設用 cp950 編碼，中文輸出被導向檔案或管線時會變亂碼。
    這裡統一改成 UTF-8，讓 `python src/eda.py > log.txt` 之類的用法也正常。
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001  失敗就維持原樣，不要讓程式掛掉
                pass


_force_utf8_stdout()

# ---------------------------------------------------------------- 路徑
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # Kaggle 原始資料解壓後放這
PROCESSED_DIR = DATA_DIR / "processed"  # 轉好的 PNG + YOLO 標註

TRAIN_IMAGE_DIR = RAW_DIR / "stage_2_train_images"
TEST_IMAGE_DIR = RAW_DIR / "stage_2_test_images"
TRAIN_LABELS_CSV = RAW_DIR / "stage_2_train_labels.csv"
DETAILED_CLASS_CSV = RAW_DIR / "stage_2_detailed_class_info.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "stage_2_sample_submission.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------- 競賽常數
COMPETITION = "rsna-pneumonia-detection-challenge"

ORIGINAL_IMAGE_SIZE = 1024          # 原始 DICOM 都是 1024x1024
IOU_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]  # 官方評分門檻

CLASS_NAMES = [
    "Normal",
    "No Lung Opacity / Not Normal",
    "Lung Opacity",
]

# ---------------------------------------------------------------- 前處理設定
IMAGE_SIZE = 640        # 訓練用解析度（RTX 4060 8GB 建議 640）
VAL_RATIO = 0.2         # 驗證集比例
RANDOM_SEED = 42

# ---------------------------------------------------------------- 訓練設定
EPOCHS = 30
BATCH_SIZE = 8          # 8GB VRAM @ 640px 的安全值；OOM 就調小
MODEL_NAME = "yolov8s.pt"

# 推論後處理（這兩個參數對分數影響最大，務必在驗證集上掃描）
CONF_THRESHOLD = 0.35   # 信心門檻：太低會因為「無框影像亂畫框」而大量扣分
NMS_IOU = 0.45
BOX_SHRINK = 0.90       # 預測框縮放係數，社群經驗值 0.85~0.90


def ensure_dirs() -> None:
    """建立所有輸出資料夾（已存在就略過）。

    在 Kaggle 上 RAW_DIR 會被指到 /kaggle/input/...，那是**唯讀**掛載點。
    對一個唯讀檔案系統上的既有資料夾呼叫 mkdir，作業系統回的是
    OSError(errno 30, Read-only file system)，而不是 FileExistsError ——
    所以 exist_ok=True 擋不住，會直接讓整支程式掛掉。
    既然資料夾已經在那裡了，本來就不需要建立：先檢查存在，再試著建。
    """
    for d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        if d.exists():
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # 建不起來的是輸入資料夾就無所謂；真正要寫檔的地方建不起來，
            # 後面存圖時自然會報錯，不用在這裡提前中斷。
            print(f"[config] 無法建立 {d}（{exc.strerror}），略過。")


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    for name in ("RAW_DIR", "PROCESSED_DIR", "FIGURES_DIR", "MODELS_DIR"):
        print(f"{name:14s} = {globals()[name]}")
