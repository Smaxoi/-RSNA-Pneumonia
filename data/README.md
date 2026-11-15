# data/ 資料夾說明

⚠️ **這個資料夾裡的資料檔不會進 git**（見 `.gitignore`），每個人都要自己下載一份。

## 資料夾結構

```
data/
├── raw/                              ← Kaggle 原始資料（約 4 GB）
│   ├── stage_2_train_images/         26,684 個 .dcm
│   ├── stage_2_test_images/          3,000 個 .dcm
│   ├── stage_2_train_labels.csv
│   ├── stage_2_detailed_class_info.csv
│   └── stage_2_sample_submission.csv
└── processed/                        ← prepare_data.py 產生（約 2 GB）
    ├── images/{train,val,test}/*.png
    ├── labels/{train,val}/*.txt      YOLO 格式標註
    ├── split.csv                     train/val 切分表
    └── data.yaml                     Ultralytics 設定檔
```

## 怎麼取得

```bash
python src/download_data.py
```

需要先設定 Kaggle API Token，步驟見 [../docs/01_環境建置.md](../docs/01_環境建置.md)。

## 硬碟空間

| 項目 | 大小 |
|---|---|
| 下載的 zip | ~3.7 GB |
| 解壓後的 raw/ | ~4.0 GB |
| processed/（640px PNG） | ~1.5 GB |
| **建議預留** | **12 GB** |

下載解壓完成後，`data/raw/*.zip` 可以刪掉省空間。

## 想省事的替代方案

如果有人電腦空間不夠或網路太慢，可以直接在 **Kaggle Notebook** 或 **Google Colab** 上跑：
資料在 Kaggle Notebook 裡已經掛在 `/kaggle/input/rsna-pneumonia-detection-challenge/`，
不用下載。只要把 `src/config.py` 裡的 `RAW_DIR` 改成那個路徑即可。
