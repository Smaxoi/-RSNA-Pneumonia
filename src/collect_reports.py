#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把群組裡的回報行整理成簡報 P4 的比較表。

用法：
    1. 把群組裡所有回報行貼進一個純文字檔（一行一筆，中間夾雜別的聊天內容沒關係）
    2. python src/collect_reports.py reports/回報行.txt

輸出：
    - Round 0 一致性檢查表（含 checksum 核對與 ±0.02 判定）
    - Round 1 變因比較表（可直接貼進 docs/04_P4比較表.md 和簡報）

這支程式不依賴 config.py，可以單獨拿去 Kaggle 或別人的電腦上跑。
"""
import re
import sys

TOL = 0.02  # 容許誤差：同設定的兩個人差在這個範圍內算一致


def _utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 每個欄位各自抓，缺哪個就是 None —— 舊版回報行欄位比較少，也要能吃
PATTERNS = {
    "name":        r"^\s*\[([^\]]+)\]",
    "round":       r"\b(round[01])\b",
    "model":       r"\b(yolov8[nsmlx])\b",
    "imgsz":       r"imgsz=(\d+)",
    "ep":          r"\bep=(\d+)",
    "mosaic":      r"mosaic=([\d.]+)",
    "batch":       r"batch=(\d+)",
    "seed":        r"seed=(\d+)",
    "fixed_conf":  r"mAP@conf([\d.]+)=",
    "fixed_map":   r"(?:mAP@conf[\d.]+|val mAP)=([\d.]+)",
    "best_map":    r"best=([\d.]+)\(",
    "best_conf":   r"best=[\d.]+\(conf=([\d.]+)",
    "best_shrink": r"shrink=([\d.]+)\)",
    "maxconf":     r"maxconf=([\d.]+)",
    "checksum":    r"checksum=([0-9a-f]+)",
    "note":        r"//\s*(.+?)\s*$",
}


def parse_line(line):
    """認得出是回報行就回傳 dict，否則回傳 None。"""
    if "[" not in line or "->" not in line:
        return None
    row = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, line)
        row[key] = m.group(1) if m else None
    if row["name"] is None or row["fixed_map"] is None:
        return None
    row["raw"] = line.strip()
    return row


def f(row, key, fmt="{}"):
    v = row.get(key)
    return "—" if v is None else fmt.format(v)


def score(row):
    """這一筆真正的成績：有掃描就用 best，沒有就只能用固定門檻那個。"""
    if row["best_map"] is not None:
        return float(row["best_map"]), "best"
    return float(row["fixed_map"]), "fixed"


def round0_table(rows):
    print("## Round 0 一致性檢查（五個人同設定）")
    print()
    print("| 人 | 固定門檻 mAP | best | 最佳 conf | shrink | maxconf | checksum |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| {} | {} | {} | {} | {} | {} | `{}` |".format(
            r["name"],
            f(r, "fixed_map"),
            f(r, "best_map"),
            f(r, "best_conf"),
            f(r, "best_shrink"),
            f(r, "maxconf"),
            r["checksum"] or "缺",
        ))
    print()

    # checksum 必須全對上，不然後面所有比較都無效
    sums = {r["checksum"] for r in rows if r["checksum"]}
    if len(sums) > 1:
        print("- ❌ **checksum 不一致**：{} —— val split 不同，這些數字不能互相比。".format(
            "、".join(sorted(sums))))
        print("  有人改了 SEED 或 SUBSET_PATIENTS，先修好再重跑。")
    elif len(sums) == 1:
        # 只報「已回報的這幾個人」，不要講成五個人都對上了 ——
        # 沒回報的人可能正拿著不同的 split，說成全員一致會掩蓋這件事。
        n = sum(1 for r in rows if r["checksum"])
        print("- ✅ checksum 一致（`{}`）—— 已回報的 {} 人用的是同一份 val split。".format(
            sums.pop(), n))
        if n < 5:
            print("  還有 {} 人沒回報，**還不能說全員對齊**。".format(5 - n))

    vals = [score(r)[0] for r in rows]
    kinds = {score(r)[1] for r in rows}
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        label = "best" if kinds == {"best"} else "各自可取得的最高分數"
        verdict = "✅ 一致" if spread <= TOL else "❌ 超出容許誤差"
        print("- {}：{}的全距 = **{:.4f}**（容許 ±{}）".format(verdict, label, spread, TOL))
        if spread > TOL:
            print("  差最多的兩個人要一起看 CONFIG 和 YOLO 標註檔。")
    if all(float(r["fixed_map"]) == 0.0 for r in rows):
        print("- ⚠️ 所有人固定門檻下都是 0.0000。這**不算通過一致性檢查** ——")
        print("  管線壞掉的人也會拿到 0.0000。請改看 best 那一欄。")
    print()


def round1_table(rows, baselines):
    print("## Round 1 變因比較（簡報 P4 的主表）")
    print()
    print("| 人 | 變因 | 模型 | imgsz | ep | mosaic | best mAP | 最佳 conf | shrink | 對 Round 0 的差距 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        best, _ = score(r)
        _, kind = score(r)
        base = baselines.get(r["name"])
        if base is None:
            delta = "— (無 Round 0)"
        elif base[1] != kind:
            # 一邊是 best、一邊只有固定門檻，相減沒有意義
            delta = "— (基準需重跑)"
        else:
            delta = "{:+.4f}".format(best - base[0])
        print("| {} | {} | {} | {} | {} | {} | **{:.4f}** | {} | {} | {} |".format(
            r["name"],
            r["note"] or "（未註明）",
            f(r, "model"),
            f(r, "imgsz"),
            f(r, "ep"),
            f(r, "mosaic"),
            best,
            f(r, "best_conf"),
            f(r, "best_shrink"),
            delta,
        ))
    print()
    if rows:
        win = max(rows, key=lambda r: score(r)[0])
        print("- 目前最好的設定：**{}** 的 `{}`，best mAP = **{:.4f}**".format(
            win["name"], win["note"] or "（未註明）", score(win)[0]))
        print("- 提醒：差距在 ±{} 以內的兩個設定**不能說誰比較好**，那是雜訊。".format(TOL))
    print()


def main(argv):
    _utf8_stdout()
    if len(argv) > 1:
        with open(argv[1], encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    rows = [r for r in (parse_line(ln) for ln in text.splitlines()) if r]
    if not rows:
        print("沒有讀到任何回報行。")
        print("回報行長這樣（Notebook 第 6 節會自己印出來）：")
        print("[姓名] round0 yolov8n imgsz=512 ep=10 mosaic=1.0 batch=16 seed=42 "
              "-> mAP@conf0.1=0.1778 best=0.1862(conf=0.05,shrink=1.0) "
              "maxconf=0.187 checksum=2f22fdec3b1c")
        return 1

    r0 = [r for r in rows if r["round"] == "round0"]
    r1 = [r for r in rows if r["round"] == "round1"]
    other = [r for r in rows if r["round"] is None]

    print("<!-- 由 src/collect_reports.py 產生，共讀到 {} 筆回報行 -->".format(len(rows)))
    print()
    if r0:
        round0_table(r0)
    baselines = {r["name"]: score(r) for r in r0}
    if r1:
        round1_table(r1, baselines)
    if other:
        print("## 看不出是哪一輪的回報行（{} 筆）".format(len(other)))
        print()
        for r in other:
            print("- `{}`".format(r["raw"]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
