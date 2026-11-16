"""
RSNA Pneumonia Detection 官方評分指標的本地實作。

為什麼要自己寫一份？
  競賽用的是自訂 metric（不是 COCO mAP），而且有兩個很特別的規則：
    1. 單張影像分數的分母是 TP + FP + FN（一般 precision 只有 TP + FP），
       所以「漏框」和「多框」一樣會扣分。
    2. 若某張影像「沒有 ground truth 框」卻畫了框 -> 該影像直接 0 分。
       若「沒有 GT 也沒有預測」-> 這張影像不計入平均（回傳 None）。

  有了本地 metric，就能在驗證集上直接掃信心門檻，不用等 Kaggle 的 late submission。

用法：
    from metric import map_iou, mean_average_precision, sweep_confidence

所有 box 格式一律為 [x, y, width, height]（左上角 + 寬高），單位是原始 1024 像素。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import IOU_THRESHOLDS  # noqa: E402
except ImportError:
    # 單獨下載這一支檔案時（例如 Kaggle Notebook）沒有 config.py，
    # 用官方門檻當後備值。這兩份數值必須永遠一致。
    IOU_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

Box = Sequence[float]


# --------------------------------------------------------------------- IoU
def iou(box_a: Box, box_b: Box) -> float:
    """兩個 [x, y, w, h] 框的 Intersection over Union。"""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ------------------------------------------------------------- 單張影像分數
def map_iou(
    boxes_true: Sequence[Box],
    boxes_pred: Sequence[Box],
    scores: Sequence[float] | None = None,
    thresholds: Sequence[float] = tuple(IOU_THRESHOLDS),
) -> float | None:
    """
    計算單張影像的 average precision。

    回傳
    ----
    float : 0.0 ~ 1.0 的分數
    None  : GT 與預測皆為空 —— 依官方規則這張影像不計入平均
    """
    boxes_true = list(boxes_true)
    boxes_pred = list(boxes_pred)

    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    # GT 空但有預測 -> 全是 FP -> 0 分（官方特別強調的規則）
    if len(boxes_true) == 0:
        return 0.0
    # 有 GT 但沒預測 -> 全是 FN -> 0 分
    if len(boxes_pred) == 0:
        return 0.0

    # 依信心值由高到低排序：高信心的框優先被用來配對
    if scores is not None:
        order = np.argsort(np.asarray(scores, dtype=float))[::-1]
        boxes_pred = [boxes_pred[i] for i in order]

    total = 0.0
    for t in thresholds:
        matched_pred: set[int] = set()
        tp = 0
        for bt in boxes_true:
            for j, bp in enumerate(boxes_pred):
                if j in matched_pred:
                    continue
                if iou(bt, bp) >= t:
                    matched_pred.add(j)
                    tp += 1
                    break
        fn = len(boxes_true) - tp
        fp = len(boxes_pred) - len(matched_pred)
        denom = tp + fn + fp
        total += tp / denom if denom else 0.0

    return total / len(thresholds)


# --------------------------------------------------------------- 整體分數
def mean_average_precision(
    gt_by_image: dict[str, Sequence[Box]],
    pred_by_image: dict[str, tuple[Sequence[Box], Sequence[float]]],
    thresholds: Sequence[float] = tuple(IOU_THRESHOLDS),
) -> dict[str, float]:
    """
    參數
    ----
    gt_by_image   : {patientId: [box, ...]}，沒有框就給空 list
    pred_by_image : {patientId: ([box, ...], [score, ...])}

    回傳
    ----
    {"mAP": 平均分數, "n_scored": 計分張數, "n_skipped": 被略過的張數}
    """
    scores_per_image: list[float] = []
    skipped = 0

    for pid, gt_boxes in gt_by_image.items():
        pred_boxes, pred_scores = pred_by_image.get(pid, ([], []))
        s = map_iou(gt_boxes, pred_boxes, pred_scores, thresholds)
        if s is None:
            skipped += 1
        else:
            scores_per_image.append(s)

    mean = float(np.mean(scores_per_image)) if scores_per_image else 0.0
    return {"mAP": mean, "n_scored": len(scores_per_image), "n_skipped": skipped}


# ------------------------------------------------------------- 信心門檻掃描
def sweep_confidence(
    gt_by_image: dict[str, Sequence[Box]],
    raw_pred_by_image: dict[str, tuple[Sequence[Box], Sequence[float]]],
    thresholds: Iterable[float] = np.arange(0.05, 0.96, 0.05),
    box_shrink: float = 1.0,
) -> list[tuple[float, float]]:
    """
    掃描信心門檻，找出讓 mAP 最高的設定。
    這是本競賽 CP 值最高的調參步驟，請務必在驗證集上跑過。

    box_shrink : 對預測框做等比例縮放（以中心為基準），社群經驗 0.85~0.90 有幫助。

    回傳 [(conf, mAP), ...]
    """
    results = []
    for conf in thresholds:
        filtered: dict[str, tuple[list[Box], list[float]]] = {}
        for pid, (boxes, scores) in raw_pred_by_image.items():
            keep_b, keep_s = [], []
            for b, s in zip(boxes, scores):
                if s < conf:
                    continue
                keep_b.append(shrink_box(b, box_shrink) if box_shrink != 1.0 else b)
                keep_s.append(s)
            filtered[pid] = (keep_b, keep_s)
        res = mean_average_precision(gt_by_image, filtered)
        results.append((round(float(conf), 3), res["mAP"]))
    return results


def shrink_box(box: Box, factor: float) -> list[float]:
    """以框中心為基準等比例縮放。"""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * factor, h * factor
    return [cx - nw / 2, cy - nh / 2, nw, nh]


# --------------------------------------------------- submission 格式工具
def boxes_to_prediction_string(boxes: Sequence[Box], scores: Sequence[float]) -> str:
    """[(x,y,w,h)], [conf] -> 'conf x y w h conf x y w h ...'"""
    parts: list[str] = []
    for (x, y, w, h), s in zip(boxes, scores):
        parts.append(f"{s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}")
    return " ".join(parts)


def parse_prediction_string(s: str | float) -> tuple[list[list[float]], list[float]]:
    """'conf x y w h ...' -> ([[x,y,w,h], ...], [conf, ...])"""
    if not isinstance(s, str) or not s.strip():
        return [], []
    nums = [float(v) for v in s.split()]
    if len(nums) % 5 != 0:
        raise ValueError(f"PredictionString 長度不是 5 的倍數: {s!r}")
    boxes, scores = [], []
    for i in range(0, len(nums), 5):
        conf, x, y, w, h = nums[i:i + 5]
        scores.append(conf)
        boxes.append([x, y, w, h])
    return boxes, scores


# ------------------------------------------------------------------ 自我測試
if __name__ == "__main__":
    print("metric.py 自我測試")
    print("-" * 46)

    # 完全重合 -> IoU = 1
    assert abs(iou([0, 0, 100, 100], [0, 0, 100, 100]) - 1.0) < 1e-9
    # 完全不重疊 -> IoU = 0
    assert iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0
    print("[ OK ] iou()")

    # 完美預測
    assert map_iou([[0, 0, 100, 100]], [[0, 0, 100, 100]], [1.0]) == 1.0
    # 沒有 GT 卻畫框 -> 0 分
    assert map_iou([], [[0, 0, 100, 100]], [1.0]) == 0.0
    # 兩邊都空 -> 不計分
    assert map_iou([], [], []) is None
    # 有 GT 但沒預測 -> 0 分
    assert map_iou([[0, 0, 100, 100]], [], []) == 0.0
    print("[ OK ] map_iou()")

    gt = {"a": [[0, 0, 100, 100]], "b": [], "c": [[10, 10, 50, 50]]}
    pred = {
        "a": ([[0, 0, 100, 100]], [0.9]),   # 完美 -> 1.0
        "b": ([], []),                       # 不計分
        "c": ([[200, 200, 50, 50]], [0.8]),  # 完全錯 -> 0.0
    }
    res = mean_average_precision(gt, pred)
    assert res["n_scored"] == 2 and res["n_skipped"] == 1
    assert abs(res["mAP"] - 0.5) < 1e-9
    print(f"[ OK ] mean_average_precision() -> {res}")

    s = boxes_to_prediction_string([[0, 0, 100, 100]], [0.5])
    b, sc = parse_prediction_string(s)
    assert b == [[0.0, 0.0, 100.0, 100.0]] and sc == [0.5]
    assert parse_prediction_string("") == ([], [])
    print(f"[ OK ] submission 格式轉換 -> {s!r}")

    print("-" * 46)
    print("全部通過。")
