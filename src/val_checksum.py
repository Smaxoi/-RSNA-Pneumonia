#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""驗證集 checksum：五個人比對「是不是同一份 val split」的唯一標準。

為什麼需要這支程式
------------------
docs/02 只用文字描述算法（「patientId 排序後取 md5 前 12 碼」）。
文字描述不夠精確：排序方式、分隔符號、編碼、結尾有沒有換行，
任何一個不同都會得到完全不同的 12 碼。五個人各自手寫 md5 必然對不起來，
而對不起來會被誤判成「切分不同」，然後去修一個根本沒壞的東西。

所以算法必須只有一份實作，就是這裡的 val_checksum()。

用法
----
# 1) 已經有 val 的 patientId 清單（一行一個，順序無所謂）
python src/val_checksum.py reports/val_ids.txt

# 2) 直接從 config.py 的設定重新切一次（需要原始 CSV）
python src/val_checksum.py --from-split

# 3) 自我檢查：確認這支程式在你的環境上行為正確
python src/val_checksum.py --selftest
"""
import hashlib
import sys

DIGITS = 12   # 取 md5 前 12 碼
SEP = "|"     # 分隔符。**必須和 notebooks/round0_kaggle_baseline.ipynb
              # 第 5 格一致**，那裡寫的是 "|".join(val_sorted)。
              # 用別的字元會得到完全不同的 12 碼。


def val_checksum(patient_ids) -> str:
    """把一組 patientId 轉成 12 碼 checksum。

    規則全部釘死，不留任何解釋空間：
      1. 去重（同一個 patientId 只算一次）
      2. 以字串升冪排序 —— 所以呼叫者傳進來的順序不影響結果
      3. 用 SEP（"|"）串起來，不加結尾分隔符
      4. utf-8 編碼後取 md5，取前 12 個十六進位字元

    這四步跟 notebooks/round0_kaggle_baseline.ipynb 第 5 格逐字對應。
    兩邊必須永遠一致，否則同一份 val split 會算出兩個不同的 checksum。
    """
    uniq = sorted({str(p).strip() for p in patient_ids if str(p).strip()})
    return hashlib.md5(SEP.join(uniq).encode("utf-8")).hexdigest()[:DIGITS]


def _selftest() -> int:
    """不需要資料也能跑，確認算法在這台機器上行為一致。"""
    a = ["b", "a", "c"]
    b = ["c", "b", "a"]           # 只有順序不同
    c = ["a", "b", "c", "a"]      # 多一個重複值
    s = val_checksum(a)

    checks = [
        ("順序不影響結果", val_checksum(b) == s),
        ("重複值不影響結果", val_checksum(c) == s),
        ("長度是 12", len(s) == DIGITS),
        ("只有十六進位字元", all(ch in "0123456789abcdef" for ch in s)),
        ("空白會被忽略", val_checksum([" a ", "b", "c", ""]) == s),
        ("內容不同就會不同", val_checksum(["a", "b", "d"]) != s),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n  三個值 a/b/c 的 checksum = {s}")
    return 0 if ok else 1


def _from_split() -> int:
    """用 config.py 的設定重新切一次，印出 checksum 和用到的參數。"""
    import pandas as pd
    import config
    from prepare_data import make_split

    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    classes = pd.read_csv(config.DETAILED_CLASS_CSV)
    per_image = make_split(labels, classes)
    val_ids = per_image.loc[per_image["split"] == "val", "patientId"]

    print("用到的參數（任何一個不同，checksum 就會不同）：")
    print(f"  RANDOM_SEED     = {config.RANDOM_SEED}")
    print(f"  VAL_RATIO       = {config.VAL_RATIO}")
    print(f"  SUBSET_PATIENTS = {getattr(config, 'SUBSET_PATIENTS', None)}")
    print(f"  抽樣後影像數    = {len(per_image)}")
    print()
    print(f"n_val = {len(val_ids)}")
    print(f"val checksum = {val_checksum(val_ids)}")
    print()
    print("⚠️  拿這個值跟 docs/02 記的 2f22fdec3b1c 核對：")
    print("    一樣   -> 基準確立，repo 終於可以重現這個值。")
    print("    不一樣 -> notebook 第 2 格的抽樣方式跟這裡不同。")
    print("    二選一當基準並更新 docs/02，不要讓兩個值同時存在。")
    return 0


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if "--selftest" in argv:
        return _selftest()
    if "--from-split" in argv:
        return _from_split()

    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 1

    ids = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            ids += [line.strip() for line in fh if line.strip()]

    uniq = {i for i in ids}
    print(f"讀到 {len(ids)} 行，去重後 {len(uniq)} 個 patientId")
    print(f"val checksum = {val_checksum(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
