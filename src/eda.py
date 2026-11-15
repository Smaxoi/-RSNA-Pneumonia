"""
RSNA Pneumonia Detection — 探索式資料分析（EDA）

一次產出 Week 12 簡報第 3 頁需要的所有圖表與統計。
輸出位置：
    reports/eda_summary.txt      文字統計（基本結構、類別分布、數值摘要）
    reports/figures/01_*.png     類別分布（長條圖）
    reports/figures/02_*.png     框尺寸分布（直方圖 + 箱型圖）
    reports/figures/03_*.png     DICOM metadata 分布
    reports/figures/04_*.png     特徵與預測目標的相關性熱力圖
    reports/figures/05_*.png     框中心點空間熱力圖
    reports/figures/06_*.png     影像範例（含 ground truth 框）

用法：
    python src/eda.py                     # 預設抽 3000 張讀 DICOM metadata
    python src/eda.py --sample-size 8000  # 想要更準就抽多一點（會比較慢）
    python src/eda.py --sample-size 0     # 全部讀（約 26,684 張，數分鐘）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不開視窗，直接存檔
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# 中文字型（Windows 用微軟正黑體；找不到會自動往後退）
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
    "Noto Sans CJK TC", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
sns.set_theme(style="whitegrid", rc={"font.sans-serif": plt.rcParams["font.sans-serif"]})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ====================================================================== 載入
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.TRAIN_LABELS_CSV.exists():
        print(f"[FAIL] 找不到 {config.TRAIN_LABELS_CSV}")
        print("       請先執行： python src/download_data.py")
        raise SystemExit(1)

    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    classes = pd.read_csv(config.DETAILED_CLASS_CSV)
    return labels, classes


def build_per_image(labels: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """把「每個框一列」整理成「每張影像一列」。"""
    per_image = (
        labels.groupby("patientId")
        .agg(n_boxes=("x", lambda s: int(s.notna().sum())),
             Target=("Target", "max"))
        .reset_index()
    )
    cls = classes.drop_duplicates(subset="patientId")[["patientId", "class"]]
    return per_image.merge(cls, on="patientId", how="left")


# ================================================================ 文字摘要
def write_summary(labels: pd.DataFrame, classes: pd.DataFrame,
                  per_image: pd.DataFrame, meta: pd.DataFrame | None) -> None:
    out = config.REPORTS_DIR / "eda_summary.txt"
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f" {title}")
        lines.append("=" * 70)

    section("1. 資料基本結構")
    lines.append(f"stage_2_train_labels.csv  shape = {labels.shape}  (列 = 每個 bounding box)")
    lines.append(f"獨立影像數 (unique patientId) = {labels['patientId'].nunique():,}")
    lines.append("")
    lines.append("欄位型別：")
    lines.append(labels.dtypes.to_string())
    lines.append("")
    lines.append("缺失值數量（Target=0 的影像沒有框，所以 x/y/width/height 為 NaN）：")
    lines.append(labels.isna().sum().to_string())
    lines.append("")
    lines.append("前 8 列範例：")
    lines.append(labels.head(8).to_string(index=False))
    lines.append("")
    lines.append("有框的範例（Target=1）：")
    lines.append(labels[labels.Target == 1].head(6).to_string(index=False))

    section("2. 類別分布")
    cls_counts = classes.drop_duplicates("patientId")["class"].value_counts()
    lines.append("每張影像的細分類別（stage_2_detailed_class_info.csv）：")
    for name, cnt in cls_counts.items():
        lines.append(f"  {name:<32s} {cnt:7,}  ({cnt / cls_counts.sum() * 100:5.2f}%)")
    lines.append("")
    tgt = per_image["Target"].value_counts().sort_index()
    lines.append("二元標籤 Target（以影像計）：")
    for k, v in tgt.items():
        lines.append(f"  Target = {k}  {v:7,}  ({v / tgt.sum() * 100:5.2f}%)")
    lines.append("")
    lines.append("每張影像的框數分布：")
    lines.append(per_image["n_boxes"].value_counts().sort_index().to_string())

    section("3. Bounding box 數值分布")
    boxes = labels.dropna(subset=["x"]).copy()
    boxes["area"] = boxes["width"] * boxes["height"]
    boxes["aspect"] = boxes["width"] / boxes["height"]
    boxes["cx"] = boxes["x"] + boxes["width"] / 2
    boxes["cy"] = boxes["y"] + boxes["height"] / 2
    lines.append(f"總框數 = {len(boxes):,}")
    lines.append("")
    lines.append(boxes[["x", "y", "width", "height", "area", "aspect", "cx", "cy"]]
                 .describe().round(2).to_string())
    lines.append("")
    lines.append(f"框面積佔整張影像 ({config.ORIGINAL_IMAGE_SIZE}x{config.ORIGINAL_IMAGE_SIZE}) 的比例："
                 f"平均 {boxes['area'].mean() / config.ORIGINAL_IMAGE_SIZE**2 * 100:.2f}%")

    if meta is not None and not meta.empty:
        section("4. DICOM Metadata")
        lines.append(f"取樣張數 = {len(meta):,}")
        lines.append("")
        lines.append("年齡：")
        lines.append(meta["age"].describe().round(2).to_string())
        lines.append("")
        lines.append("性別分布：")
        lines.append(meta["sex"].value_counts().to_string())
        lines.append("")
        lines.append("拍攝視角 ViewPosition 分布：")
        lines.append(meta["view"].value_counts().to_string())
        lines.append("")
        lines.append("ViewPosition x Target 交叉表（列百分比）：")
        ct = pd.crosstab(meta["view"], meta["Target"], normalize="index") * 100
        lines.append(ct.round(2).to_string())
        lines.append("")
        lines.append(">>> 若 AP 的 Target=1 比例明顯高於 PA，代表拍攝視角是有用的輔助特徵。")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ OK ] 文字摘要 -> {out}")


# ================================================================== 圖 1
def fig_class_distribution(per_image: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    order = [c for c in config.CLASS_NAMES if c in per_image["class"].unique()]
    sns.countplot(data=per_image, y="class", order=order, ax=axes[0],
                  hue="class", palette=PALETTE[:len(order)], legend=False)
    axes[0].set_title("細分類別分布（每張影像）", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("影像張數")
    axes[0].set_ylabel("")
    total = len(per_image)
    for p in axes[0].patches:
        w = p.get_width()
        axes[0].text(w + total * 0.01, p.get_y() + p.get_height() / 2,
                     f"{int(w):,}\n({w / total * 100:.1f}%)", va="center", fontsize=9)
    axes[0].set_xlim(0, total * 0.62)

    sns.countplot(data=per_image, x="Target", ax=axes[1],
                  hue="Target", palette=[PALETTE[0], PALETTE[3]], legend=False)
    axes[1].set_title("二元標籤 Target 分布", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Target（0 = 無肺炎證據，1 = 有）")
    axes[1].set_ylabel("影像張數")
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].text(p.get_x() + p.get_width() / 2, h + total * 0.01,
                     f"{int(h):,}\n({h / total * 100:.1f}%)", ha="center", fontsize=10)
    axes[1].set_ylim(0, total * 0.85)

    box_counts = per_image["n_boxes"].value_counts().sort_index()
    axes[2].bar(box_counts.index.astype(str), box_counts.values, color=PALETTE[2])
    axes[2].set_title("每張影像的 bounding box 數量", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("框數")
    axes[2].set_ylabel("影像張數")
    axes[2].set_yscale("log")
    for i, (k, v) in enumerate(box_counts.items()):
        axes[2].text(i, v * 1.1, f"{v:,}", ha="center", fontsize=9)

    fig.suptitle("圖 1 ─ 類別分布（資料不平衡：約 2/3 影像沒有框）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_class_distribution.png")


# ================================================================== 圖 2
def fig_bbox_size(labels: pd.DataFrame) -> None:
    b = labels.dropna(subset=["x"]).copy()
    b["area_ratio"] = b["width"] * b["height"] / config.ORIGINAL_IMAGE_SIZE**2 * 100
    b["aspect"] = b["width"] / b["height"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    sns.histplot(b["width"], bins=50, ax=axes[0, 0], color=PALETTE[0])
    axes[0, 0].set_title("框寬度 width 分布")
    axes[0, 0].set_xlabel("像素")

    sns.histplot(b["height"], bins=50, ax=axes[0, 1], color=PALETTE[1])
    axes[0, 1].set_title("框高度 height 分布")
    axes[0, 1].set_xlabel("像素")

    sns.histplot(b["area_ratio"], bins=50, ax=axes[0, 2], color=PALETTE[2])
    axes[0, 2].set_title("框面積佔整圖比例")
    axes[0, 2].set_xlabel("%")

    sns.boxplot(data=b[["width", "height"]], ax=axes[1, 0], palette=PALETTE[:2], hue=None)
    axes[1, 0].set_title("寬 / 高 箱型圖")
    axes[1, 0].set_ylabel("像素")

    sns.histplot(b["aspect"], bins=50, ax=axes[1, 1], color=PALETTE[3])
    axes[1, 1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1, 1].set_title("長寬比 width / height（虛線 = 正方形）")
    axes[1, 1].set_xlabel("比值")

    axes[1, 2].scatter(b["width"], b["height"], s=3, alpha=0.08, color=PALETTE[0])
    axes[1, 2].set_title("寬 vs 高 散佈圖")
    axes[1, 2].set_xlabel("width (px)")
    axes[1, 2].set_ylabel("height (px)")

    fig.suptitle("圖 2 ─ Bounding Box 尺寸分布（框普遍偏高瘦，符合肺葉形狀）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_bbox_size_distribution.png")


# ====================================================== DICOM metadata
def read_metadata(per_image: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """從 DICOM header 抽出年齡 / 性別 / 拍攝視角。"""
    try:
        import pydicom
    except ImportError:
        print("[WARN] 沒有 pydicom，跳過 metadata 分析")
        return pd.DataFrame()

    if not config.TRAIN_IMAGE_DIR.is_dir():
        print(f"[WARN] 找不到影像資料夾 {config.TRAIN_IMAGE_DIR}，跳過 metadata 分析")
        return pd.DataFrame()

    df = per_image
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=config.RANDOM_SEED)

    rows = []
    for pid, target, n_boxes in tqdm(
        zip(df["patientId"], df["Target"], df["n_boxes"]),
        total=len(df), desc="讀取 DICOM metadata", ncols=80,
    ):
        path = config.TRAIN_IMAGE_DIR / f"{pid}.dcm"
        if not path.exists():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            age_raw = str(getattr(ds, "PatientAge", "")).strip()
            age = int(age_raw) if age_raw.isdigit() else np.nan
            rows.append({
                "patientId": pid,
                "age": age,
                "sex": str(getattr(ds, "PatientSex", "")).strip() or "Unknown",
                "view": str(getattr(ds, "ViewPosition", "")).strip() or "Unknown",
                "Target": int(target),
                "n_boxes": int(n_boxes),
            })
        except Exception:  # noqa: BLE001  單張讀取失敗不要中斷整個 EDA
            continue

    return pd.DataFrame(rows)


# ================================================================== 圖 3
def fig_metadata(meta: pd.DataFrame) -> None:
    if meta.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.histplot(data=meta[meta["age"].between(0, 100)], x="age", hue="Target",
                 bins=40, multiple="stack", ax=axes[0],
                 palette=[PALETTE[0], PALETTE[3]])
    axes[0].set_title("病人年齡分布（依 Target 堆疊）")
    axes[0].set_xlabel("年齡")
    axes[0].set_ylabel("張數")

    sns.countplot(data=meta, x="sex", hue="Target", ax=axes[1],
                  palette=[PALETTE[0], PALETTE[3]])
    axes[1].set_title("性別 x Target")
    axes[1].set_xlabel("性別")

    ct = pd.crosstab(meta["view"], meta["Target"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=axes[2],
            color=[PALETTE[0], PALETTE[3]])
    axes[2].set_title("拍攝視角 ViewPosition 的肺炎比例")
    axes[2].set_xlabel("ViewPosition（AP 多為臥床重症）")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=0)
    for i, view in enumerate(ct.index):
        if 1 in ct.columns:
            axes[2].text(i, 50, f"{ct.loc[view, 1]:.1f}%", ha="center",
                         color="white", fontweight="bold")

    fig.suptitle(f"圖 3 ─ DICOM Metadata 分布（取樣 {len(meta):,} 張）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_metadata_distribution.png")


# ================================================================== 圖 4
def fig_correlation(meta: pd.DataFrame) -> None:
    if meta.empty:
        return
    df = pd.DataFrame({
        "年齡": meta["age"],
        "男性(sex=M)": (meta["sex"] == "M").astype(int),
        "視角AP(view=AP)": (meta["view"] == "AP").astype(int),
        "框數": meta["n_boxes"],
        "肺炎Target": meta["Target"],
    }).dropna()

    corr = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"label": "Pearson 相關係數"}, ax=ax)
    ax.set_title("圖 4 ─ 特徵與預測目標的相關性熱力圖",
                 fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    _save(fig, "04_correlation_heatmap.png")

    if "肺炎Target" in corr:
        print("\n  與 Target 的相關性：")
        for k, v in corr["肺炎Target"].drop("肺炎Target").sort_values(
                key=abs, ascending=False).items():
            print(f"    {k:<18s} {v:+.3f}")


# ================================================================== 圖 5
def fig_spatial_heatmap(labels: pd.DataFrame) -> None:
    b = labels.dropna(subset=["x"]).copy()
    cx = b["x"] + b["width"] / 2
    cy = b["y"] + b["height"] / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    S = config.ORIGINAL_IMAGE_SIZE
    h = axes[0].hist2d(cx, cy, bins=64, range=[[0, S], [0, S]], cmap="inferno")
    axes[0].set_title("框中心點密度", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].set_aspect("equal")
    fig.colorbar(h[3], ax=axes[0], label="框數")

    # 把所有框疊起來，得到「哪些像素最常被框到」
    mask = np.zeros((S // 8, S // 8), dtype=np.float32)
    for x, y, w, hgt in b[["x", "y", "width", "height"]].to_numpy():
        x0, y0 = int(x // 8), int(y // 8)
        x1, y1 = int((x + w) // 8), int((y + hgt) // 8)
        mask[y0:y1, x0:x1] += 1
    im = axes[1].imshow(mask, cmap="inferno")
    axes[1].set_title("框覆蓋次數疊加圖", fontweight="bold")
    axes[1].set_xlabel("x (/8 px)")
    axes[1].set_ylabel("y (/8 px)")
    fig.colorbar(im, ax=axes[1], label="覆蓋次數")

    fig.suptitle("圖 5 ─ 肺炎陰影的空間分布（左右肺雙叢集，是很強的位置先驗）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_bbox_spatial_heatmap.png")


# ================================================================== 圖 6
def fig_samples(labels: pd.DataFrame, per_image: pd.DataFrame, n_each: int = 3) -> None:
    try:
        import pydicom
    except ImportError:
        print("[WARN] 沒有 pydicom，跳過影像範例圖")
        return
    if not config.TRAIN_IMAGE_DIR.is_dir():
        print("[WARN] 找不到影像資料夾，跳過影像範例圖")
        return

    rng = np.random.default_rng(config.RANDOM_SEED)
    picks: list[tuple[str, str]] = []
    for cls in config.CLASS_NAMES:
        pool = per_image[per_image["class"] == cls]["patientId"].to_numpy()
        if len(pool) == 0:
            continue
        for pid in rng.choice(pool, size=min(n_each, len(pool)), replace=False):
            picks.append((str(pid), cls))

    if not picks:
        return

    cols = n_each
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (pid, cls) in zip(axes, picks):
        path = config.TRAIN_IMAGE_DIR / f"{pid}.dcm"
        if not path.exists():
            ax.axis("off")
            continue
        ds = pydicom.dcmread(str(path))
        ax.imshow(ds.pixel_array, cmap="gray")

        boxes = labels[(labels.patientId == pid) & labels.x.notna()]
        for _, r in boxes.iterrows():
            ax.add_patch(Rectangle((r.x, r.y), r.width, r.height,
                                   fill=False, edgecolor="#FF3B30", linewidth=2.2))
        view = str(getattr(ds, "ViewPosition", "?"))
        age = str(getattr(ds, "PatientAge", "?"))
        sex = str(getattr(ds, "PatientSex", "?"))
        ax.set_title(f"{cls}\n{len(boxes)} 個框 | {view} | {age}歲 {sex}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle("圖 6 ─ 影像範例（紅框 = ground truth；注意第 2 列「不正常但非肺炎」也有白影）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_sample_images.png")


# =================================================================== 工具
def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"[ OK ] {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RSNA 競賽資料 EDA")
    parser.add_argument("--sample-size", type=int, default=3000,
                        help="讀取 DICOM metadata 的取樣張數，0 = 全部（預設 3000）")
    parser.add_argument("--skip-images", action="store_true",
                        help="跳過所有需要讀 DICOM 的分析（只做 CSV 分析）")
    args = parser.parse_args()

    config.ensure_dirs()

    print("載入 CSV ...")
    labels, classes = load_data()
    per_image = build_per_image(labels, classes)
    print(f"  標註列數 = {len(labels):,}，獨立影像數 = {len(per_image):,}")

    print("\n產生圖表 ...")
    fig_class_distribution(per_image)
    fig_bbox_size(labels)
    fig_spatial_heatmap(labels)

    meta = pd.DataFrame()
    if not args.skip_images:
        meta = read_metadata(per_image, args.sample_size)
        fig_metadata(meta)
        fig_correlation(meta)
        fig_samples(labels, per_image)

    write_summary(labels, classes, per_image, meta if not meta.empty else None)

    print("\n" + "=" * 62)
    print(f" 完成！圖表在 {config.FIGURES_DIR}")
    print(f"       統計在 {config.REPORTS_DIR / 'eda_summary.txt'}")
    print(" 簡報第 3 頁直接拿這些圖，記得在該頁附上 Kaggle Code 來源 URL。")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
