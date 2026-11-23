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
# 下面這些值是 **Round 0 的共同基準**，不是「建議值」。
# 五個人必須用完全一樣的設定跑完 Round 0，數字才能互相比較。
# 要試自己的變因請等 Round 1，而且一次只改一個。
IMAGE_SIZE = 512        # Round 0 固定 512（本機 8GB VRAM 跑 640 也可以，但那是 Round 1 的變因）
VAL_RATIO = 0.2         # 驗證集比例
RANDOM_SEED = 42

# 只取這麼多個病人來跑，控制訓練時間。**改了它 checksum 就會變。**
SUBSET_PATIENTS = 8000  # None = 用全部 26,684 張

# ---------------------------------------------------------------- 訓練設定
EPOCHS = 10             # Round 0 固定 10；延長訓練是 Round 1 的變因
BATCH_SIZE = 16
MODEL_NAME = "yolov8n.pt"   # Round 0 固定 n；n/s/m 比較是古晏慈的 Round 1 變因

# 推論後處理（這兩個參數對分數影響最大，務必在驗證集上掃描）
#
# ⚠️ CONF_THRESHOLD 踩過的坑：原本預設 0.35，而 Round 0 的模型最高只給到
#    0.187 的信心 —— 門檻高於模型的最高信心，所有框被濾光，五個人全部拿到
#    mAP = 0.0000。**先看模型的 maxconf，再決定門檻**，不要沿用別人的經驗值。
CONF_THRESHOLD = 0.10   # Round 0 的固定回報門檻；實際最佳值要掃描才知道
NMS_IOU = 0.45
BOX_SHRINK = 1.0        # Round 0 不縮框；縮放係數是林明和的 Round 1 變因


# ---------------------------------------------------------------- Round 0 基準
# 凍結的對照表：跑 Round 0 之前用 assert_round0() 確認自己沒改到東西。
# 這裡刻意寫死數值而不是引用上面的變數 —— 上面被改掉時這裡才抓得出來。
ROUND0 = {
    "IMAGE_SIZE": 512,
    "VAL_RATIO": 0.2,
    "RANDOM_SEED": 42,
    "SUBSET_PATIENTS": 8000,
    "EPOCHS": 10,
    "BATCH_SIZE": 16,
    "MODEL_NAME": "yolov8n.pt",
    "CONF_THRESHOLD": 0.10,
    "BOX_SHRINK": 1.0,
}


def assert_round0(strict: bool = False) -> list:
    """檢查目前設定是否還是 Round 0 的基準，回傳不一致的項目清單。

    strict=True 時只要有一項不同就丟 RuntimeError。
    影響 checksum 的三個參數（RANDOM_SEED、VAL_RATIO、SUBSET_PATIENTS）
    會特別標出來 —— 那三個不同的話，連 val split 都不是同一份。
    """
    affects_checksum = {"RANDOM_SEED", "VAL_RATIO", "SUBSET_PATIENTS"}
    diffs = []
    for key, expected in ROUND0.items():
        actual = globals().get(key)
        if actual != expected:
            diffs.append((key, expected, actual, key in affects_checksum))

    if not diffs:
        print("[config] 目前設定符合 Round 0 基準。")
        return []

    print("[config] ⚠️ 與 Round 0 基準不一致：")
    for key, expected, actual, is_checksum in diffs:
        mark = "  ← 這個會讓 checksum 對不起來" if is_checksum else ""
        print(f"  {key}: 基準 {expected!r} / 目前 {actual!r}{mark}")
    if strict:
        raise RuntimeError("設定與 Round 0 基準不一致，見上方清單。")
    return diffs


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
