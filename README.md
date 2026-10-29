# 完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型

本專案以安大叔於 2021 年 6 月 9 日發布的[指定文章](https://zhuanlan.zhihu.com/p/379291795)為主要依據，實作 **PyTorch、MNIST 手寫數字與 784→32→10 的單隱藏層神經網路**。依照原文的 11 步主線，學習選擇 GPU、搬移資料與模型、訓練及驗證，再加入獨立測試、模型保存與實驗證據。

實作設定另與 Adwait Dathan 的[英文原文](https://www.analyticsvidhya.com/blog/2021/05/a-complete-hands-on-guide-to-train-your-neural-network-model-on-google-colab-gpu/)及文末連結的[作者 notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)核對。原文與本專案的逐步對照、必要修正和補強，集中於 [SOURCE_ALIGNMENT.md](docs/SOURCE_ALIGNMENT.md)。

**已在這台 RTX 4060 Laptop GPU 完成 10 輪 MNIST 訓練，獨立測試 accuracy 為 96.99%（9,699／10,000 張），訓練與驗證計時 43.48 秒。** [實驗摘要](results/mnist_rtx4060/summary.md)、[完整報告](docs/REPORT.md)及[已訓練模型](results/mnist_rtx4060/best_model.pt)均已保存。本次數字來自本機 Windows 筆電；Colab 雲端尚未另行實測，以下保留完整操作指南。

## 目錄

- [學習目標與專案內容](#學習目標與專案內容)
- [為什麼神經網路會使用 GPU](#為什麼神經網路會使用-gpu)
- [依原文完成 11 步實作](#依原文完成-11-步實作)
- [模型、資料與訓練設定](#模型資料與訓練設定)
- [實驗成果與檔案](#實驗成果與檔案)
- [本機 RTX 4060 執行方式](#本機-rtx-4060-執行方式)
- [常見問題](#常見問題)
- [GitHub 整理與重做實驗](#github-整理與重做實驗)
- [來源與參考資料](#來源與參考資料)

## 學習目標與專案內容

完成本教學後，應能解釋並操作：

1. Colab 的 GPU 執行階段與 PyTorch 實際裝置檢查。
2. `get_default_device`、`to_device`、`DeviceDataLoader` 的分工。
3. MNIST 的訓練／驗證切分與單隱藏層分類模型。
4. SGD、交叉熵、反向傳播與兩階段學習率。
5. 驗證模型選擇、獨立測試、儲存與還原模型。

```text
README.md                         依指定文章整理的完整操作指南
notebooks/colab_gpu_training.ipynb 依 11 步編排、可獨立上傳 Colab 的筆記本
src/training.py                   MNIST、MLP、裝置搬移、訓練與輸出核心
train.py                          本機命令列入口
requirements.txt                  一般套件需求
requirements-gpu.txt              本機 CUDA 實驗的固定版本
tests/                            重要訓練行為檢查
docs/REPORT.md                    期中報告正文與實驗分析
docs/SOURCE_ALIGNMENT.md          來源、11 步對照及補強說明
docs/EXPERIMENTS.md               實驗設定與證據解讀
results/mnist_rtx4060/            本機 MNIST GPU 實測成果
```

筆記本不依賴預先建立的 GitHub 儲存庫網址，直接上傳即可操作。

## 為什麼神經網路會使用 GPU

神經網路會反覆執行矩陣乘法與其他張量運算。GPU 能同時處理大量相似運算，適合支援這類平行工作；CPU 仍負責程式控制、資料準備等工作。GPU 是整個訓練流程的一部分，並不取代 CPU 的所有用途。這也是[指定文章](https://zhuanlan.zhihu.com/p/379291795)先介紹 GPU 再進入實作的核心觀念。

程式真正使用 GPU，需要同時做到三件事：**取得可用的 GPU、將模型移到 GPU、將每批輸入與標籤移到同一 GPU**。只在 Colab 選取硬體加速器，或只看到電腦配有顯示卡，都不足以確認 PyTorch 已使用它。

小模型也不保證 GPU 一定比 CPU 快；資料傳輸、批次大小與執行開銷會影響時間。本教學的目標是正確完成 GPU 訓練流程，不預先承諾加速倍數。

## 依原文完成 11 步實作

以下對應原文的步驟順序。完整程式在[筆記本](notebooks/colab_gpu_training.ipynb)，共用核心在 [src/training.py](src/training.py)。

### 步驟 1：建立 Colab 筆記本並選擇 GPU

1. 開啟 [Google Colab](https://colab.research.google.com/)。
2. 選擇「檔案 → 上傳筆記本」（File → Upload notebook），上傳 `notebooks/colab_gpu_training.ipynb`。
3. 選擇「執行階段 → 變更執行階段類型」（Runtime → Change runtime type），將硬體加速器設為可用的 GPU。
4. 從第一格依序執行，或選擇「全部執行」（Run all）。

原文提到的 Tesla K80 與當年的使用時數不應當作目前保證。現行 Colab GPU 型號、額度與可用時間會變動，請以實際環境輸出為準。[Colab 官方 FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)

### 步驟 2：載入必要套件

筆記本載入 PyTorch、torchvision、NumPy 與 Matplotlib，並顯示 Python、PyTorch、CUDA 與 GPU 名稱。Colab 請先使用預裝的相容 PyTorch／torchvision，不任意重裝 CUDA 套件。

### 步驟 3：建立訓練與驗證資料

沿用原文的 `MNIST` 與 `ToTensor()`：將 28×28 灰階影像轉成 0～1 的張量。官方訓練資料再切成 **50,000 張訓練與 10,000 張驗證**。本報告新增 seed 42，固定這個隨機切分。

另載入 `train=False` 的官方 10,000 張測試資料，留到模型選定後評估；獨立測試是本報告的補強。`train`、`transform` 與 `download` 的功能可參考 [torchvision MNIST 官方說明](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。

### 步驟 4：用 DataLoader 建立批次

沿用訓練 batch size **128**、驗證 batch size **256**；測試亦使用 256。訓練打亂順序，驗證與測試不打亂。本機 Windows 使用 `num_workers=0`，Colab 可選用 2。

使用 CUDA 時可啟用 `pin_memory`，配合後續 `non_blocking=True` 搬移。它們協助資料傳輸，不保證在所有工作負載下都有相同加速效果。

### 步驟 5：建立 MnistModel

將每張影像展平成 784 個值，經過 32 個隱藏神經元與 ReLU，再輸出 10 個類別分數（logits）。標籤代表數字 0～9。

模型直接將 logits 交給交叉熵，不在損失計算前額外加入 Softmax。預測時取最高分的類別。[PyTorch 損失與最佳化教學](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

### 步驟 6：選擇 GPU 或 CPU

`get_default_device()` 檢查 `torch.cuda.is_available()`，有 CUDA 時選擇 GPU，否則使用 CPU。這保留原文的裝置選擇概念。

本報告另提供「必須使用 GPU」選項，避免默默退回 CPU 後仍把結果寫成 GPU 實驗。筆記本的 `REQUIRE_GPU = True` 或命令列 `--require-gpu` 都會在沒有 GPU 時停止。若無 GPU 仍想練習，可將筆記本的 `REQUIRE_GPU` 設為 False，允許 CPU 備援；有 CUDA 時仍優先使用 GPU，實際裝置以輸出為準。

### 步驟 7：定義 to_device

`to_device(data, device)` 將張量送到選定裝置；當輸入是一批影像與標籤組成的 list／tuple，便逐項搬移。模型也能透過相同概念移至 GPU。`non_blocking=True` 允許適當條件下的非同步搬移，實際效果仍受資料與裝置環境影響。

### 步驟 8：建立 DeviceDataLoader

`DeviceDataLoader` 包住原本的 DataLoader。每次迭代先取出一批資料，再透過 `to_device` 搬至指定裝置，讓訓練與驗證都能取得放在正確位置的影像、標籤。

### 步驟 9：定義訓練與驗證流程

每批資料依序進行前向傳播、計算交叉熵、反向傳播與 SGD 更新，並清除梯度。每輪結束後驗證，記錄 loss、accuracy 與當輪學習率。

本報告修正原文展示程式的缺漏：明確使用 `model.train()`／`model.eval()`、驗證停用梯度，並按樣本數加權彙總指標，避免最後一個小批次被等權計算。這些修正不改變原文的模型與任務。

### 步驟 10：建立模型並搬至 GPU

建立 `MnistModel` 後，將模型放到與 DeviceDataLoader 相同的裝置。若資料位於 GPU、模型仍在 CPU，運算會因裝置不一致而失敗。筆記本會顯示裝置，讓這一步可以檢查。

### 步驟 11：執行兩階段訓練與驗證

[作者 notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)的兩次訓練呼叫為先 **5 輪 SGD、learning rate 0.5**，再以同一模型續訓 **5 輪、learning rate 0.1**。本專案依此設定完整的 10 輪流程，切換階段時重建無 momentum 的 SGD。

筆記本預設旗標如下，請保留該格後續建立 `CONFIG` 的程式：

```python
REQUIRE_GPU = True
QUICK = False
EPOCHS = 10
```

快速模式將 `QUICK` 改為 True，只用 **2,048／512／512 張、1 輪**，用於確認程式，不是正式實驗。

訓練前先記錄未訓練模型的驗證指標；完成 10 輪後，載入驗證 loss 最低的權重，對獨立測試集評估一次。本報告另提供模型還原、單張推論、Drive 保存與 ZIP 下載。變更參數後，需從整個參數格開始依序重跑資料、模型與訓練。

## 模型、資料與訓練設定

| 階段 | 輸出形狀 |
| --- | --- |
| 輸入影像 | `N × 1 × 28 × 28` |
| 展平 | `N × 784` |
| Linear 784→32 | `N × 32` |
| ReLU | `N × 32` |
| Linear 32→10 | `N × 10` logits |

`N` 是當批影像數。可訓練參數共 **25,450**：第一層 `784×32+32`，第二層 `32×10+10`。

| 項目 | 完整模式 |
| --- | --- |
| 資料／前處理 | MNIST／僅 ToTensor，沒有額外 Normalize |
| 訓練／驗證／測試 | 50,000／10,000／10,000 |
| 訓練／驗證與測試 batch size | 128／256 |
| 模型／隱藏神經元 | MLP／32 |
| optimizer／loss | 無 momentum SGD／CrossEntropyLoss |
| 學習率 | 第 1～5 輪 0.5；第 6～10 輪 0.1 |
| seed／Windows workers | 42／0 |
| 最佳 checkpoint | 驗證 loss 最低 |

## 實驗成果與檔案

本次在 RTX 4060 完整執行，以下數字可由 [metrics.json](results/mnist_rtx4060/metrics.json) 與 [history.csv](results/mnist_rtx4060/history.csv) 核對：

| 指標 | 實測結果 |
| --- | ---: |
| 未訓練模型驗證 accuracy | 9.86% |
| 最佳驗證 epoch | 10 |
| 最佳驗證 loss／accuracy | 0.114256／96.57% |
| 最終測試 loss／accuracy | 0.100092／**96.99%** |
| 測試正確／錯誤張數 | 9,699／301 |
| Macro F1 | 0.969714 |
| 訓練與驗證計時 | 43.48 秒 |

![MNIST 在 RTX 4060 上的實際學習曲線](results/mnist_rtx4060/training_curves.png)

前五輪使用 learning rate 0.5，第六輪起為 0.1。驗證 loss 曾在第 3、5 輪上升，後段最低值位於第 10 輪，因此選擇該輪模型測試。完整觀察與限制見 [REPORT.md](docs/REPORT.md)。

![MNIST 測試混淆矩陣](results/mnist_rtx4060/confusion_matrix.png)

例如 16 張真實數字 5 被判為 3，14 張數字 4 被判為 9；錯誤數量來自 [confusion_matrix.csv](results/mnist_rtx4060/confusion_matrix.csv)。更多錯例與模型載入說明見 [實驗摘要](results/mnist_rtx4060/summary.md)。以下檔案保留同一輪實驗的證據：

| 檔案 | 用途 |
| --- | --- |
| `config.json`、`environment.json` | 設定、套件與實際裝置 |
| `initial_validation.json` | 未訓練模型的驗證指標 |
| `history.csv` | 每輪 loss、accuracy、learning rate 與時間 |
| `metrics.json` | 最終測試、最佳 epoch、每類指標等 |
| `best_model.pt` | 最低驗證 loss 的已訓練權重 |
| `training_curves.png` | 學習曲線 |
| `confusion_matrix.png`、`confusion_matrix.csv` | 真實數字與預測數字的分類統計 |
| `predictions.png`、`misclassified.png` | 預測與錯例；無錯誤時不產生錯例圖 |
| `gpu_runtime.json`、`checkpoint_verification.json` | GPU 執行證據與模型載入檢查 |

`test_accuracy` 為 0～1 的比例，顯示百分比時乘以 100。混淆矩陣的列／縱軸是真實數字，欄／橫軸是預測數字。每類 precision、recall、F1 與樣本數，可補充整體 accuracy 未呈現的差異。

## 本機 RTX 4060 執行方式

本機實驗使用 NVIDIA GeForce RTX 4060 Laptop GPU、Python 3.12.14、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、CUDA runtime 12.8。安裝與完整執行：

```bash
python -m pip install -r requirements-gpu.txt
python train.py --require-gpu --output-dir outputs/mnist_repeat
```

一般環境與 CPU 備援可使用：

```bash
python -m pip install -r requirements.txt
python train.py
```

程式預設跑完整 10 輪；快速檢查為：

```bash
python train.py --quick
```

每次使用新的輸出資料夾，避免混用設定與成果。執行核心行為檢查：

```bash
python -m unittest discover -s tests -v
```

## 常見問題

| 問題 | 處理方式 |
| --- | --- |
| CUDA 不可用 | Colab 先確認硬體加速器與環境輸出；本機確認相容驅動與 CUDA 版 PyTorch。若刻意改用 CPU，需標明裝置。 |
| 模型與資料裝置不同 | 確認步驟 7～10 均已依序執行，模型和每批影像、標籤都送到同一裝置。 |
| CUDA out of memory | 降低 batch size，從參數格開始依序重跑；記錄此變更，不把新設定與基準混在一起。 |
| DataLoader 在 Windows 無法啟動 | 使用預設 `num_workers=0`；Colab 可再嘗試 2。 |
| 下載失敗 | 確認網路並稍後重試；不要把資料下載失敗當成模型訓練結果。 |
| 改了參數卻沒有生效 | 重新執行整個設定格，再重建資料與模型；只改變文字不會更新已存在的物件。 |
| Colab 重連後成果不見 | 從先前下載 ZIP 或 Drive 取回；暫存虛擬機器會被回收。 |
| 數字與文章不同 | 原文未固定 seed，本報告也修正指標彙總；裝置及套件版本不同，不應期待逐位相同。 |

## GitHub 整理與重做實驗

上傳時保留 README、程式、筆記本、依賴、測試、報告、來源對照與本次 MNIST 結果。不要上傳 `.venv/`、資料集快取與無關私人檔案。已訓練的小型模型可連同成果保存，方便還原推論。

本報告的 GitHub 儲存庫為 [korykoryko66/mitest](https://github.com/korykoryko66/mitest)。更新時使用「Add file → Upload files」，將上傳資料夾裡的檔案與子資料夾放在儲存庫根目錄，保留結構，再確認文件、筆記本與成果圖片可以開啟。重做實驗時，以新的資料夾保存設定、環境與結果；具體原則見 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

Colab 虛擬機器會因閒置或生命週期結束被刪除，保存筆記本不等於保存模型與所有輸出。請使用筆記本提供的 Drive 或 ZIP 步驟。[Colab 檔案保存說明](https://research.google.com/colaboratory/intl/en-GB/faq.html)

## 來源與參考資料

1. [安大叔：指定知乎文章](https://zhuanlan.zhihu.com/p/379291795)，2021-06-09，本文主要依據。
2. [Adwait Dathan：英文原文](https://www.analyticsvidhya.com/blog/2021/05/a-complete-hands-on-guide-to-train-your-neural-network-model-on-google-colab-gpu/)，2021-05-28。
3. [原作者 GitHub notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)：兩階段學習率與可核對程式。
4. [torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。
5. [PyTorch 最佳化流程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。
6. [PyTorch 模型保存與載入](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。
7. [Google Colab 官方 FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)。
