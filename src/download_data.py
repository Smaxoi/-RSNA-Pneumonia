"""
用 Kaggle API 下載競賽資料並解壓到 data/raw/。

前置作業（每位組員只要做一次）：
  1. 登入 Kaggle，然後在終端機執行：  kaggle auth login
     （新版 Kaggle CLI 2.2+ 走 OAuth，會開瀏覽器讓你授權，之後自動快取）
  2. 到競賽頁面 https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
     按下 "I Understand and Accept"（沒接受規則會下載失敗，回傳 403）

  舊版 token 方式仍可用（擇一即可）：
    * 環境變數 KAGGLE_API_TOKEN=<token>
    * ~/.kaggle/access_token
    * ~/.kaggle/kaggle.json（最舊的格式）

用法：
    python src/download_data.py              # 下載 + 解壓
    python src/download_data.py --no-unzip   # 只下載 zip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# 競賽檔案總量約 3.7 GB，解壓後約 4.0 GB，請確認硬碟空間足夠
EXPECTED_FILES = [
    "stage_2_train_labels.csv",
    "stage_2_detailed_class_info.csv",
    "stage_2_sample_submission.csv",
    "stage_2_train_images.zip",
    "stage_2_test_images.zip",
]


def check_credentials() -> bool:
    """
    問 Kaggle CLI 自己有沒有通過認證（OAuth 快取 / 環境變數 / token 檔都算）。
    注意：只看回傳碼，不印出 token 內容。
    """
    import os

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "kaggle", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0:
            print("[ OK ] Kaggle 認證已完成")
            return True
    except Exception:  # noqa: BLE001  舊版 CLI 沒有 auth 子指令，往下走 fallback
        pass

    # 舊版 CLI 的 fallback：檢查檔案 / 環境變數
    legacy = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".kaggle" / "access_token",
    ]
    for p in legacy:
        if p.exists():
            print(f"[ OK ] 找到 Kaggle 憑證: {p}")
            return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("[ OK ] 使用環境變數 KAGGLE_API_TOKEN")
        return True

    print("[FAIL] 還沒有 Kaggle 認證。")
    print()
    print("  最簡單的做法（新版 CLI）：")
    print("     kaggle auth login")
    print("  會開瀏覽器讓你授權，完成後憑證自動快取，不用管檔案放哪。")
    print()
    print("  替代做法：到 https://www.kaggle.com/settings/api 產生 token，然後擇一")
    print(f"     * 存成 {Path.home() / '.kaggle' / 'access_token'}")
    print("     * 或設環境變數 KAGGLE_API_TOKEN")
    print()
    print("  ⚠️ 另外別忘了到競賽頁面按 'I Understand and Accept' 接受規則，")
    print("     否則下載會回 403。")
    return False


def download(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", config.COMPETITION,
        "-p", str(dest),
    ]
    print(f"\n執行： {' '.join(cmd)}")
    print("（約 3.7 GB，依網速可能要 10~40 分鐘）\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[FAIL] 下載失敗。最常見的原因：")
        print("  * 403 Forbidden -> 還沒在競賽頁面接受規則")
        print("  * 401 Unauthorized -> kaggle.json 位置或內容不對")
        return False
    return True


def unzip_all(raw_dir: Path) -> None:
    """解開外層 zip，再解開裡面的 stage_2_*_images.zip。"""
    outer = sorted(raw_dir.glob("*.zip"))
    if not outer:
        print("[WARN] 找不到任何 zip 檔，可能已經解壓過了。")
        return

    for zpath in outer:
        # 影像 zip 要解到同名資料夾，其餘直接解到 raw/
        if zpath.stem in ("stage_2_train_images", "stage_2_test_images"):
            target = raw_dir / zpath.stem
        else:
            target = raw_dir

        if target.is_dir() and any(target.glob("*.dcm")):
            print(f"[SKIP] {zpath.name} 已解壓過 -> {target}")
            continue

        print(f"解壓 {zpath.name} -> {target} ...")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)
        print(f"[ OK ] {zpath.name} 完成")

    # 外層 zip 解開後可能還有內層影像 zip
    for name in ("stage_2_train_images", "stage_2_test_images"):
        inner = raw_dir / f"{name}.zip"
        target = raw_dir / name
        if inner.exists() and not any(target.glob("*.dcm")):
            print(f"解壓 {inner.name} -> {target} ...")
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(target)


def summarize(raw_dir: Path) -> None:
    print("\n" + "=" * 56)
    print(" data/raw 內容")
    print("=" * 56)
    for name in ("stage_2_train_images", "stage_2_test_images"):
        d = raw_dir / name
        if d.is_dir():
            print(f"  {name:<28s} {sum(1 for _ in d.glob('*.dcm')):,} 個 .dcm")
    for name in EXPECTED_FILES:
        if name.endswith(".csv"):
            f = raw_dir / name
            status = f"{f.stat().st_size / 1024**2:.1f} MB" if f.exists() else "缺少"
            print(f"  {name:<28s} {status}")
    print("\n下一步： python src/eda.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="下載 RSNA 競賽資料")
    parser.add_argument("--no-unzip", action="store_true", help="只下載不解壓")
    args = parser.parse_args()

    config.ensure_dirs()

    if config.TRAIN_LABELS_CSV.exists() and any(config.TRAIN_IMAGE_DIR.glob("*.dcm")):
        print("[ OK ] 資料看起來已經齊全了，不重複下載。")
        summarize(config.RAW_DIR)
        return 0

    if not check_credentials():
        return 1
    if not download(config.RAW_DIR):
        return 1
    if not args.no_unzip:
        unzip_all(config.RAW_DIR)
    summarize(config.RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
