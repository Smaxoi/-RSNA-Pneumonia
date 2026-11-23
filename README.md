# RSNA Pneumonia Detection Challenge — 深度學習期末專題

> 競賽連結：<https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge>

## 📖 先看哪一份？

| 你是 | 先看這份 |
|---|---|
| **第一次接觸這個專題的組員** | 👉 **[docs/00_專題講解.md](docs/00_專題講解.md)** ← 白話版，15 分鐘看完 |
| 要動手裝環境、跑程式 | [docs/01_環境建置.md](docs/01_環境建置.md) |
| 想知道誰負責什麼、時程 | [docs/02_分工與時程.md](docs/02_分工與時程.md) |
| **已經認領了變因，想知道自己要做什麼** | 👉 **[docs/03_任務卡.md](docs/03_任務卡.md)** ← 五張卡，只讀自己那張 |
| **要寫 P3 那頁（資料分析）** | 👉 **[docs/05_P3簡報文案.md](docs/05_P3簡報文案.md)** ← 圖已經跑好了，結論句可直接貼 |
| **要整理初步實驗結果（簡報 P4）** | 👉 **[docs/04_P4比較表.md](docs/04_P4比較表.md)** ← 回報行丟進去就產出表格 |
| **要跑 Round 0 共用 baseline** | 👉 **[notebooks/round0_kaggle_baseline.ipynb](notebooks/round0_kaggle_baseline.ipynb)** ← 匯入 Kaggle 直接跑，不用裝環境 |
| 要查技術細節（本文件） | 繼續往下看 ↓ |

⚠️ **Week 12 只要交 5 頁簡報**，不用把整個專題做完。細節看 `00_專題講解.md`。

---

本文件是完整技術參考，內容比較密。如果只是想先搞懂專題在做什麼，請看 `00_專題講解.md`。

---

## 0. 三十秒摘要

| 項目 | 內容 |
|---|---|
| 主辦 | 北美放射學會 RSNA（2018，Featured Competition，獎金 $30,000） |
| 任務類型 | **物件偵測（Object Detection）**，不是單純分類 |
| 輸入 | 胸部 X 光片（DICOM 格式，1024×1024 灰階） |
| 輸出 | 每張影像「肺部陰影（lung opacity）」的 bounding box + 信心值 |
| 評分 | mAP over IoU thresholds 0.4 ~ 0.75（step 0.05），**自訂 metric** |
| 訓練資料 | `stage_2_train_labels.csv` + `stage_2_train_images/`（約 26,684 張 DICOM，~3.6 GB） |
| 測試資料 | `stage_2_test_images/`（3,000 張 DICOM） |
| 目前狀態 | 競賽已結束，但開放 **Late Submission**，仍可提交看分數 |

---

## 1. 這個競賽到底在做什麼？（對應簡報第 1 頁：介紹競賽任務）

### 1.1 醫學背景

肺炎（Pneumonia）在全球是兒童死亡的主因之一，佔 5 歲以下兒童死亡的 15% 以上；在美國每年造成超過 50 萬次急診就醫、5 萬人死亡。

診斷肺炎主要靠**胸部 X 光（CXR）**：肺炎在影像上表現為「**局部增加的不透明區域（increased opacity）**」——也就是一塊比周圍更白、更模糊的區域。

但這件事很難：
- 肺水腫、出血、肺塌陷、肺癌、放療/手術後變化、肋膜積水…… **都會造成類似的白影**，難以區分。
- 病人姿勢、吸氣深度不同，影像長相就會變。
- 放射科醫師每班要看大量影像，容易疲勞。

所以 RSNA 想做的是 **自動化的初篩（imaging screening）**：讓演算法先找出「疑似肺炎」的片子與位置，幫助醫師排序、加速判讀。

### 1.2 技術定義

> **給一張胸腔 X 光，標出所有「疑似肺炎的肺部陰影」的矩形框。**

- 有框 → 模型認為這張片子有肺炎證據
- 沒框 → 模型認為沒有確切肺炎證據

所以它同時包含了「**要不要給框（分類）**」與「**框在哪裡（定位）**」兩件事。
⚠️ 這點很關鍵，見第 3 節的評分機制陷阱。

### 1.3 資料從哪來

來自美國 NIH Clinical Center 公開的 ChestX-ray8 資料集，由 RSNA 與 Society of Thoracic Radiology 的放射科醫師重新標註 bounding box（MD.ai 提供標註工具）。

---

## 2. 資料長什麼樣子？（對應簡報第 2 頁：舉例說明資料的長相）

### 2.1 檔案清單

```
stage_2_train_images/            # 訓練影像資料夾，26,684 個 .dcm 檔
stage_2_test_images/             # 測試影像資料夾，3,000 個 .dcm 檔
stage_2_train_labels.csv         # 訓練標註（bounding box + Target）
stage_2_detailed_class_info.csv  # 每張影像的細分類別
stage_2_sample_submission.csv    # 繳交格式範例，含測試集的 patientId
```

### 2.2 `stage_2_train_labels.csv`

| 欄位 | 說明 |
|---|---|
| `patientId` | 影像唯一 ID（對應 `stage_2_train_images/<patientId>.dcm`） |
| `x` | bounding box 左上角 x 座標 |
| `y` | bounding box 左上角 y 座標 |
| `width` | 框寬 |
| `height` | 框高 |
| `Target` | 1 = 有肺炎證據，0 = 無 |

範例（實際跑 `python src/eda.py` 會印出真實資料）：

```
patientId                              x      y      width  height  Target
0004cfab-14fd-4e49-80ba-63a80b6bddd6   NaN    NaN    NaN    NaN     0
00436515-870c-4b36-a041-de91049b9ab4   264.0  152.0  213.0  379.0   1
00436515-870c-4b36-a041-de91049b9ab4   562.0  152.0  256.0  453.0   1
```

🔺 **重點 1：一個 `patientId` 可能有多列**（一張片子可以有多個框，通常左右肺各一個）。
🔺 **重點 2：`Target=0` 的列，x/y/width/height 都是 NaN**。
🔺 **重點 3：座標是「左上角 + 寬高」(`xywh`)，不是 `xyxy`，也不是中心點格式。**
　　→ 餵給 YOLO 前必須轉成「中心點 + 寬高，且除以影像尺寸正規化」。

### 2.3 `stage_2_detailed_class_info.csv`

| 欄位 | 說明 |
|---|---|
| `patientId` | 同上 |
| `class` | 三類之一 |

三個類別：
1. **`Normal`** — 完全正常的片子
2. **`No Lung Opacity / Not Normal`** — 不正常，但不是肺炎造成的陰影（肺水腫、積水、纖維化等）
3. **`Lung Opacity`** — 有肺炎陰影（即 `Target=1`）

🔺 類別 2 是這題的**難點**：它長得很像肺炎但答案是「沒有框」。這是最大的誤判來源，可以在簡報裡當作「資料的挑戰」來談。

### 2.4 影像本身

- 格式：**DICOM**（`.dcm`），醫學影像標準格式，用 `pydicom` 讀取
- 尺寸：**1024 × 1024**，8-bit 灰階（單通道）
- DICOM header 裡還有額外的 metadata 可以挖：
  - `PatientAge`（年齡）
  - `PatientSex`（性別）
  - `ViewPosition`（拍攝姿勢：**PA** 後前位 / **AP** 前後位）

🔺 **`ViewPosition` 是個藏寶點**：AP 位通常是躺著拍的重症病人，肺炎盛行率明顯較高。這是很好的 EDA 發現，也可以當作模型的輔助特徵。

### 2.5 讀一張 DICOM 的最小範例

```python
import pydicom
import matplotlib.pyplot as plt

ds = pydicom.dcmread("data/raw/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm")
print(ds.PatientAge, ds.PatientSex, ds.ViewPosition)  # 例：'37' 'F' 'PA'
img = ds.pixel_array          # numpy array, shape = (1024, 1024), dtype = uint8
plt.imshow(img, cmap="gray")
```

---

## 3. 怎麼評分？（這題最容易踩雷的地方）

### 3.1 公式

評分是 **mean Average Precision over IoU thresholds**。

IoU（交集 / 聯集）門檻掃過 8 個值：`0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75`

每個門檻 `t` 下，算這張影像的精確度：

$$\frac{TP(t)}{TP(t) + FP(t) + FN(t)}$$

然後對 8 個門檻取平均 → 得到**單張影像的分數**；再對所有測試影像取平均 → 最終分數。

### 3.2 三個必須知道的陷阱

> 🚨 **陷阱 1：分母含 FN。**
> 一般的 precision 分母只有 `TP + FP`，但這裡**多加了 FN**。意思是「漏掉的框」跟「多畫的框」一樣會扣分，所以**不能只衝 recall 也不能只衝 precision**。

> 🚨 **陷阱 2：沒有框的影像，亂畫一個框就直接 0 分。**
> 官方原文：若某張影像完全沒有 ground truth box，**任何數量的預測都會讓這張影像得 0 分**，而且這張仍然計入平均。
> → 由於測試集裡約 **2/3 的影像是沒有框的**，「該不該輸出框」的決策（也就是信心值門檻 threshold）對分數的影響，**往往比框畫得準不準還大**。這是本題最重要的調參項目。

> 🚨 **陷阱 3：IoU 門檻從 0.4 起跳，最高只到 0.75。**
> 比 COCO 的 0.5~0.95 寬鬆，代表「框得大概對」就能拿分，不必追求像素級精準。力氣要花在「要不要出框」而不是「框得超準」。

### 3.3 繳交格式

```
patientId,PredictionString
0004cfab-14fd-4e49-80ba-63a80b6bddd6,0.5 0 0 100 100
00313ee0-9eaa-42f4-b0ab-c148ed3241cd,
00322d4d-1c29-4943-afc9-b6754be640eb,0.8 10 10 50 50 0.75 100 100 5 5
```

- 每張影像**只能一列**，所有框寫在同一個 `PredictionString` 裡
- 每個框是 5 個數字：`confidence x y width height`（空白分隔）
- 預測「沒有肺炎」就**留空**（逗號後面什麼都不寫）
- 座標要是**原始 1024×1024 尺度**，如果你用 512 或 640 訓練，記得**乘回去**

我們已在 `src/metric.py` 實作了這個 metric，可以在本地驗證集上直接算分，不用等 Kaggle。

---

## 4. 我們要做的資料分析（對應簡報第 3 頁：做簡單的資料分析）

簡報要求的每一項，都已對應到 `src/eda.py` 裡的一段程式，跑一次就全部產出到 `reports/figures/`：

| 簡報要求 | 我們怎麼做 | 產出檔案 |
|---|---|---|
| 了解資料的基本結構（貼範例） | `df.head()` / `shape` / `dtypes` / 缺失值統計 | `reports/eda_summary.txt` |
| 類別分布（長條圖） | `class` 三類分布、`Target` 0/1 分布、每張影像的框數分布 | `01_class_distribution.png` |
| 變數的分布（箱型圖、直方圖） | 框的 width / height / 面積 / 長寬比分布 | `02_bbox_size_distribution.png` |
| 變數的分布（直方圖） | DICOM metadata：年齡分布、性別、ViewPosition | `03_metadata_distribution.png` |
| 預測目標與特徵的關係（熱力圖） | ① metadata 與 Target 的相關性熱力圖 ② 框中心點的空間熱力圖 | `04_correlation_heatmap.png`、`05_bbox_spatial_heatmap.png` |
| 舉例說明資料的長相 | 隨機抽樣影像並畫上 ground truth 框（正例/負例各幾張） | `06_sample_images.png` |

**預期會看到的結論**（跑完要自己確認數字）：
- 約 **1/3** 的影像有肺炎框，**2/3** 沒有 → 類別不平衡
- 框的面積分布很廣，但多集中在影像中段偏外側（左右肺的位置）→ 空間熱力圖會呈現雙叢集
- **AP 視角的肺炎比例明顯高於 PA 視角** → 強相關特徵
- 多數陽性影像有 **2 個框**（左右肺各一）

📌 **參考 Kaggle Code（簡報要求附上來源，請務必照抄引用）**
- 官方 Starter Kernel: <https://www.kaggle.com/code/peterchang77/exploratory-data-analysis>
- EDA 參考: <https://www.kaggle.com/code/zahaviguy/what-are-lung-opacities>
- 搜尋關鍵字：在競賽頁的 **Code** 分頁搜尋 `EDA`

---

## 5. 預期的實作方法（對應簡報第 4 頁）

### 5.1 整體流程

```
DICOM (1024×1024)
   │  src/prepare_data.py
   ├─► PNG (640×640) + YOLO 格式標註 (.txt)
   │
   ├─► 以 patientId 分組切 train/val（8:2，依 class 分層）
   │
   ▼
   模型訓練（見下方三個方案）
   │
   ▼
   後處理：NMS / WBF 合併框 + 信心門檻篩選
   │
   ▼
   src/metric.py 在本地驗證集算分 → 調門檻
   │
   ▼
   轉回 1024 尺度 → submission.csv → Kaggle Late Submission
```

### 5.2 建議的三階段方案

| 階段 | 方法 | 目的 | 預期 mAP |
|---|---|---|---|
| **Baseline** | **YOLOv8n/s**（Ultralytics），單類別 `opacity`，640px | 最快跑通全流程，當天就能有分數 | 0.12 ~ 0.16 |
| **改進** | YOLOv8m + 調信心門檻 + 資料增強（水平翻轉、亮度/對比） | 門檻掃描是 CP 值最高的一步 | 0.18 ~ 0.21 |
| **進階（擇一）** | ① **二階段**：先用 CNN（ResNet/EfficientNet）分類「有無肺炎」過濾掉負例，再對正例做偵測<br>② **Faster R-CNN / RetinaNet**（torchvision）做比較實驗<br>③ 多模型 **WBF 融合** | 做模型比較，簡報才有內容 | 0.20 ~ 0.24 |

> 📊 參考：當年競賽第一名的 private LB 約 **0.25**，所以 **0.15 以上就是很合理的期末專題成績**，不要被數字嚇到。

### 5.3 為什麼先選 YOLOv8

1. Ultralytics 一行指令就能訓練，**三天內一定能跑出初步結果**（簡報要求「有初步實驗結果佳」）
2. 內建資料增強、NMS、mAP 計算，不用自己刻
3. 8GB VRAM（RTX 4060）跑 640px、batch 8~16 沒問題
4. 之後要換 Faster R-CNN 做對照實驗也很容易，資料前處理可以共用

### 5.4 針對本題評分機制的專屬技巧

1. **信心門檻掃描**（最重要）：訓練完後，用 `src/metric.py` 在驗證集上掃 `conf` 從 0.05 到 0.95，畫出 mAP 曲線找最佳點。通常最佳門檻落在 **0.3~0.5**，比預設的 0.25 高。
2. **框稍微縮小**：社群普遍發現把預測框乘上 **0.85~0.90** 的縮放係數能提升分數（標註框普遍偏大）。
3. **把 `No Lung Opacity / Not Normal` 當成 hard negative** 特別加權，減少誤報。
4. **不要過度追求框的精準度**，IoU 門檻只到 0.75。

📌 **參考 Kaggle Code（請附上來源）**
- 第一名方案討論：<https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/discussion/70421>
- 搜尋關鍵字：在 **Code** 分頁搜尋 `YOLO`、`Mask R-CNN`、`RetinaNet`

---

## 6. 專案資料夾結構

```
深度學習期末專題/
├── README.md                  ← 你正在看的這份
├── environment.yml            ← conda 環境定義
├── requirements.txt           ← pip 套件清單
├── setup_env.ps1              ← 一鍵建環境腳本（Windows）
├── .gitignore
├── docs/
│   ├── 00_專題講解.md          ← 給組員的白話版導讀（先看這份）
│   ├── 01_環境建置.md          ← 環境安裝詳細步驟 + 常見錯誤
│   ├── 02_分工與時程.md        ← 五人分工、兩輪制、回報格式、時程
│   ├── 03_任務卡.md            ← 五張任務卡，每人一張，寫到可以照著做
│   ├── 04_P4比較表.md          ← 初步實驗結果彙總，簡報 P4 直接用
│   └── 05_P3簡報文案.md        ← P3 的三張圖與結論句，照實際數字寫好
├── data/
│   ├── raw/                   ← Kaggle 原始資料（不進 git）
│   ├── processed/             ← 轉好的 PNG + YOLO 標註（不進 git）
│   └── README.md
├── src/
│   ├── config.py              ← 所有路徑與超參數集中管理
│   ├── check_env.py           ← 檢查套件 / CUDA / 資料是否就緒
│   ├── download_data.py       ← 用 Kaggle API 下載並解壓資料
│   ├── eda.py                 ← 第 4 節的所有分析，一次產出所有圖
│   ├── prepare_data.py        ← DICOM → PNG + YOLO 標註 + train/val 切分
│   ├── metric.py              ← 競賽官方 metric 的本地實作
│   └── collect_reports.py     ← 把群組的回報行整理成 P4 的比較表
├── notebooks/
│   └── round0_kaggle_baseline.ipynb  ← 五人共用 baseline，匯入 Kaggle 直接跑
├── models/                    ← 訓練好的權重（不進 git）
└── reports/
    └── figures/               ← EDA 圖表輸出，簡報直接拿這裡的圖
```

---

## 7. 新組員上手

### 路線 A：Kaggle Notebook（推薦，不用裝任何東西）

Week 12 的 Round 0 全員走這條。不用裝 CUDA、不用下載 3.7 GB 資料。

1. 到競賽頁按 **Join Competition** 接受規則
2. Kaggle 首頁 → Create → Notebook → File → **Import Notebook**，
   上傳 [`notebooks/round0_kaggle_baseline.ipynb`](notebooks/round0_kaggle_baseline.ipynb)
3. 右側 Settings：**Accelerator** 選 GPU、**Internet** 開啟、
   **Input** 掛上 `rsna-pneumonia-detection-challenge`
4. 只改第一個格子的 `MY_NAME`，**Run All**，約 40~60 分鐘
5. 把最後印出的回報行貼到群組

為什麼五個人都要跑同一份 → 看 [docs/02_分工與時程.md](docs/02_分工與時程.md) 第 2 節。

### 路線 B：自己電腦（Round 1 想調參再用）

```bash
# ① 建立環境（第一次才要，約 10 分鐘）
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

```bash
# ② 啟動環境並檢查裝好了沒
conda activate rsna
python src/check_env.py
```

```bash
# ③ Kaggle 認證（會開瀏覽器授權，一次就好）
kaggle auth login
```

```bash
# ④ 下載資料（約 3.7 GB；記得先到競賽頁面按 I Understand and Accept）
python src/download_data.py
```

接著就可以跑 EDA：

```bash
python src/eda.py
```

> 環境裝不起來、Kaggle Token 怎麼設定 → 看 [docs/01_環境建置.md](docs/01_環境建置.md)
> 誰做哪一頁簡報、進度到哪 → 看 [docs/02_分工與時程.md](docs/02_分工與時程.md)

---

## 8. Week 12 繳交對照檢核表

| 簡報頁 | 內容 | 本文件對應章節 | 負責人 |
|---|---|---|---|
| 1 | 介紹競賽任務 | §1 + §3（評分） | 待填 |
| 2 | 舉例說明資料的長相 | §2（含 DICOM 範例圖） | 待填 |
| 3 | 簡單的資料分析（含來源） | §4 + `docs/05_P3簡報文案.md` | 待填 |
| 4 | 預期的實作方法 + 初步實驗結果（含來源） | §5 + `docs/04_P4比較表.md` | 待填 |
| 5 | 分工模式 | `docs/02_分工與時程.md` | 待填 |

✅ 兩個不能忘的扣分點：
1. **凡是參考 Kaggle Code 的內容，一定要在該頁附上 URL 來源**（簡報用紅字強調了兩次）
2. **第 4 頁最好要有初步實驗結果**（哪怕只是 YOLOv8n 跑 10 epoch 的 mAP 數字 + 一張預測圖）

---

## 9. 引用格式

> Anouk Stein, MD, Carol Wu, Chris Carr, George Shih, Jamie Dulkowski, kalpathy, Leon Chen, Luciano Prevedello, Marc Kohli, MD, Mark McDonald, Peter, Phil Culliton, RSNA AI Challenges, Safwan Halabi MD, and Tian Xia. *RSNA Pneumonia Detection Challenge*. https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge, 2018. Kaggle.
