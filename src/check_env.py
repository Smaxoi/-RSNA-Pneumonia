"""
環境健康檢查：套件、CUDA、資料是否就緒。

用法：
    python src/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OK = "[ OK ]"
NG = "[FAIL]"
WARN = "[WARN]"

# (import 名稱, 顯示名稱, 是否必要)
PACKAGES = [
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("scipy", "scipy", True),
    ("pydicom", "pydicom", True),
    ("cv2", "opencv-python", True),
    ("PIL", "Pillow", True),
    ("skimage", "scikit-image", False),
    ("matplotlib", "matplotlib", True),
    ("seaborn", "seaborn", True),
    ("sklearn", "scikit-learn", True),
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("ultralytics", "ultralytics", True),
    ("timm", "timm", False),
    ("albumentations", "albumentations", False),
    ("ensemble_boxes", "ensemble-boxes", False),
    ("tqdm", "tqdm", True),
    ("yaml", "PyYAML", True),
    ("kaggle", "kaggle", False),
]


def _version(mod) -> str:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if v:
            return str(v)
    return "?"


def check_packages() -> bool:
    print("=" * 62)
    print(" 套件檢查")
    print("=" * 62)
    all_required_ok = True
    for import_name, display_name, required in PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            print(f"{OK} {display_name:<20s} {_version(mod)}")
        except Exception as exc:  # noqa: BLE001
            tag = NG if required else WARN
            note = "（必要）" if required else "（選用，可先略過）"
            print(f"{tag} {display_name:<20s} 未安裝 {note}  -> {type(exc).__name__}")
            if required:
                all_required_ok = False
    return all_required_ok


def check_cuda() -> None:
    print()
    print("=" * 62)
    print(" GPU / CUDA 檢查")
    print("=" * 62)
    try:
        import torch
    except ImportError:
        print(f"{NG} 沒有 torch，無法檢查 CUDA")
        return

    print(f"      torch 版本        : {torch.__version__}")
    print(f"      torch 編譯的 CUDA : {torch.version.cuda}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"{OK} CUDA 可用，偵測到 {n} 張 GPU")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024**3
            print(f"      GPU {i}: {props.name}  ({vram:.1f} GB VRAM)")
            if vram < 7:
                print(f"{WARN}   VRAM 偏小，建議把 config.BATCH_SIZE 調到 4 或 IMAGE_SIZE 改 512")
    else:
        print(f"{WARN} CUDA 不可用 —— 現在是 CPU 模式，訓練會非常慢。")
        print("      解法見 docs/01_環境建置.md 的「CUDA 裝不起來」章節。")


def check_data() -> None:
    print()
    print("=" * 62)
    print(" 資料檢查")
    print("=" * 62)
    targets = [
        (config.TRAIN_LABELS_CSV, "訓練標註 CSV", "file"),
        (config.DETAILED_CLASS_CSV, "細分類別 CSV", "file"),
        (config.SAMPLE_SUBMISSION_CSV, "繳交範例 CSV", "file"),
        (config.TRAIN_IMAGE_DIR, "訓練影像資料夾", "dir"),
        (config.TEST_IMAGE_DIR, "測試影像資料夾", "dir"),
    ]
    missing = False
    for path, name, kind in targets:
        if not path.exists():
            print(f"{WARN} {name:<16s} 不存在: {path}")
            missing = True
            continue
        if kind == "dir":
            n = sum(1 for _ in path.glob("*.dcm"))
            print(f"{OK} {name:<16s} {n:,} 個 .dcm 檔")
        else:
            size_mb = path.stat().st_size / 1024**2
            print(f"{OK} {name:<16s} {size_mb:.1f} MB")

    if missing:
        print()
        print("      資料尚未下載。請執行：  python src/download_data.py")


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    print(f"直譯器  : {sys.executable}")
    print(f"專案根目錄: {config.PROJECT_ROOT}")
    print()

    packages_ok = check_packages()
    check_cuda()
    check_data()

    print()
    print("=" * 62)
    if packages_ok:
        print(" 必要套件齊全，環境可用。")
        print(" 下一步： python src/download_data.py  然後  python src/eda.py")
    else:
        print(" 有必要套件缺失，請執行： pip install -r requirements.txt")
    print("=" * 62)
    return 0 if packages_ok else 1


if __name__ == "__main__":
    config.ensure_dirs()
    raise SystemExit(main())
