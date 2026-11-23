"""
前處理：DICOM -> PNG，標註 -> YOLO 格式，並切分 train / val。

做了什麼
--------
1. 把 1024x1024 的 .dcm 轉成 IMAGE_SIZE x IMAGE_SIZE 的 .png（灰階轉三通道存檔）
2. 把 [x, y, w, h]（左上角+寬高，絕對像素）轉成 YOLO 的
   [class cx cy w h]（中心點+寬高，除以影像尺寸正規化到 0~1）
3. 依 patientId 分組、依 class 分層切 8:2 的 train / val
   （分組很重要：同一個病人的框不能同時出現在 train 和 val）
4. 產生 Ultralytics 需要的 data.yaml

輸出
----
data/processed/
├── images/train/*.png
├── images/val/*.png
├── labels/train/*.txt
├── labels/val/*.txt
├── split.csv          ← 每張影像屬於哪一組，之後算分數要用
└── data.yaml          ← 餵給 YOLO 的設定檔

用法
----
    python src/prepare_data.py                 # 轉 train（含 val 切分）
    python src/prepare_data.py --split test    # 轉 test 影像（推論用）
    python src/prepare_data.py --size 512      # 改解析度
    python src/prepare_data.py --limit 500     # 只轉 500 張，先驗證流程
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ----------------------------------------------------------- 單張影像轉換
def convert_one(args: tuple[str, str, str, int]) -> tuple[str, bool]:
    """讀一張 DICOM，resize 後存成 PNG。回傳 (patientId, 是否成功)。"""
    pid, src_dir, dst_dir, size = args
    src = Path(src_dir) / f"{pid}.dcm"
    dst = Path(dst_dir) / f"{pid}.png"
    if dst.exists():
        return pid, True
    try:
        arr = pydicom.dcmread(str(src)).pixel_array
        if arr.dtype != np.uint8:  # 保險：正規化到 0~255
            arr = arr.astype(np.float32)
            rng = arr.max() - arr.min()
            arr = ((arr - arr.min()) / rng * 255.0) if rng > 0 else np.zeros_like(arr)
            arr = arr.astype(np.uint8)
        if arr.shape[0] != size:
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), arr)
        return pid, True
    except Exception:  # noqa: BLE001
        return pid, False


def convert_images(pids: list[str], src_dir: Path, dst_dir: Path,
                   size: int, workers: int) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(pid, str(src_dir), str(dst_dir), size) for pid in pids]
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(convert_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"轉檔 -> {dst_dir.name}", ncols=80):
            pid, ok = fut.result()
            if not ok:
                failed.append(pid)
    if failed:
        print(f"[WARN] {len(failed)} 張轉檔失敗，例如：{failed[:5]}")
    return failed


# ------------------------------------------------------------- YOLO 標註
def write_yolo_labels(labels: pd.DataFrame, pids: set[str], dst_dir: Path) -> None:
    """
    YOLO 格式：每行 `class_id cx cy w h`，座標全部正規化到 0~1。
    只有一個類別，所以 class_id 固定為 0。
    沒有框的影像 -> 寫一個空的 .txt（YOLO 會當成純背景負樣本，很重要！）
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    S = float(config.ORIGINAL_IMAGE_SIZE)

    boxes = labels[labels.patientId.isin(pids) & labels.x.notna()]
    grouped: dict[str, list[str]] = {}
    for pid, x, y, w, h in boxes[["patientId", "x", "y", "width", "height"]].to_numpy():
        cx = (x + w / 2) / S
        cy = (y + h / 2) / S
        nw = w / S
        nh = h / S
        # 夾在 0~1，避免標註超出邊界導致 YOLO 報錯
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(nw, 1.0), min(nh, 1.0)
        grouped.setdefault(str(pid), []).append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    for pid in tqdm(pids, desc=f"標註 -> {dst_dir.name}", ncols=80):
        (dst_dir / f"{pid}.txt").write_text(
            "\n".join(grouped.get(pid, [])), encoding="utf-8"
        )


# ------------------------------------------------------------------ 切分
def make_split(labels: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """依 patientId 去重、依 class 分層切 train / val。"""
    per_image = (
        labels.groupby("patientId")
        .agg(n_boxes=("x", lambda s: int(s.notna().sum())), Target=("Target", "max"))
        .reset_index()
    )
    cls = classes.drop_duplicates("patientId")[["patientId", "class"]]
    per_image = per_image.merge(cls, on="patientId", how="left")

    # 先抽樣，再切 train/val —— 順序不能反，不然 val 的內容會不一樣。
    #
    # ⚠️ 這是 **repo 這邊的實作**。Round 0 的 2f22fdec3b1c 是 Kaggle notebook
    #    第 2 格跑出來的，那格程式不在版控裡，抽樣方式未必和這裡相同。
    #    第一次跑 `python src/val_checksum.py --from-split` 時要核對：
    #    對得上就沿用 2f22fdec3b1c，對不上就以這裡為準並更新 docs/02，
    #    兩個值不能同時存在。
    subset = getattr(config, "SUBSET_PATIENTS", None)
    if subset and subset < len(per_image):
        per_image, _ = train_test_split(
            per_image,
            train_size=subset,
            random_state=config.RANDOM_SEED,
            stratify=per_image["class"],   # 抽樣後三類比例維持不變
        )
        per_image = per_image.reset_index(drop=True)

    train_ids, val_ids = train_test_split(
        per_image["patientId"],
        test_size=config.VAL_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=per_image["class"],   # 三個類別在兩邊比例一致
    )
    per_image["split"] = np.where(per_image.patientId.isin(set(val_ids)), "val", "train")
    return per_image


# ------------------------------------------------------------- data.yaml
def write_data_yaml(size: int) -> Path:
    cfg = {
        "path": str(config.PROCESSED_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "opacity"},   # 單一類別
    }
    path = config.PROCESSED_DIR / "data.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    print(f"[ OK ] {path}")
    print(f"       訓練指令範例：")
    print(f"       yolo detect train data=\"{path}\" model={config.MODEL_NAME} "
          f"imgsz={size} epochs={config.EPOCHS} batch={config.BATCH_SIZE}")
    return path


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description="DICOM -> PNG + YOLO 標註")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0,
                        help="只處理前 N 張（用來快速驗證流程，0 = 全部）")
    args = parser.parse_args()

    config.ensure_dirs()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        if not config.TEST_IMAGE_DIR.is_dir():
            print(f"[FAIL] 找不到 {config.TEST_IMAGE_DIR}，請先跑 download_data.py")
            return 1
        pids = [p.stem for p in sorted(config.TEST_IMAGE_DIR.glob("*.dcm"))]
        if args.limit:
            pids = pids[:args.limit]
        print(f"測試集 {len(pids):,} 張")
        convert_images(pids, config.TEST_IMAGE_DIR,
                       config.PROCESSED_DIR / "images" / "test",
                       args.size, args.workers)
        print("\n完成。推論時請記得把預測框從 "
              f"{args.size} 尺度乘回 {config.ORIGINAL_IMAGE_SIZE} 再寫 submission。")
        return 0

    if not config.TRAIN_LABELS_CSV.exists():
        print(f"[FAIL] 找不到 {config.TRAIN_LABELS_CSV}，請先跑 download_data.py")
        return 1

    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    classes = pd.read_csv(config.DETAILED_CLASS_CSV)
    split_df = make_split(labels, classes)

    if args.limit:
        split_df = (split_df.groupby("split", group_keys=False)
                    .head(max(args.limit // 2, 1)))
        print(f"[WARN] --limit 模式：只處理 {len(split_df)} 張")

    split_path = config.PROCESSED_DIR / "split.csv"
    split_df.to_csv(split_path, index=False)
    print(f"[ OK ] 切分表 -> {split_path}")
    print(split_df.groupby(["split", "class"]).size().to_string())
    print()

    for part in ("train", "val"):
        pids = split_df.loc[split_df.split == part, "patientId"].astype(str).tolist()
        convert_images(pids, config.TRAIN_IMAGE_DIR,
                       config.PROCESSED_DIR / "images" / part,
                       args.size, args.workers)
        write_yolo_labels(labels, set(pids),
                          config.PROCESSED_DIR / "labels" / part)

    print()
    write_data_yaml(args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
