# 完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型

這份期中報告以 **PyTorch、Fashion-MNIST 與卷積神經網路（CNN）** 為實作範例，帶領初學者從 Google Colab 啟用 GPU，完成資料準備、訓練、驗證、測試、儲存模型與預測。這些工具是示範完整流程的選擇；報告主題是學會在 Colab GPU 上訓練神經網路。

**已在這台 NVIDIA GeForce RTX 4060 Laptop GPU 完成完整資料的 5 輪訓練，測試準確率為 89.40%，訓練與驗證計時 48.91 秒。** [GPU 實驗摘要](results/local_rtx4060/summary.md) 收錄環境、指標、圖表與 [訓練完成的模型](results/local_rtx4060/best_model.pt)。本次數字來自本機 Windows 筆電；Colab 雲端尚未另行實測，以下保留完整的 Colab 操作指南。

| RTX 4060 完整實驗 | 實測結果 |
| --- | ---: |
| 訓練／驗證／測試資料 | 54,000／6,000／10,000 張 |
| 最佳驗證 epoch | 5 |
| 測試 accuracy | **89.40%（8,940／10,000）** |
| 測試 loss／Macro F1 | 0.289209／0.894313 |
| 訓練與驗證時間 | 48.91 秒 |

完整分析見 [期中報告](docs/REPORT.md)；先前 CPU 紀錄保留於 [成果目錄](results/README.md)，作為次要參考。

## 目錄

- [學習目標](#學習目標)
- [專案內容](#專案內容)
- [從 Colab 開始](#從-colab-開始)
- [資料與前處理](#資料與前處理)
- [神經網路如何學習](#神經網路如何學習)
- [CNN 模型架構](#cnn-模型架構)
- [訓練與評估設定](#訓練與評估設定)
- [成果檔案與解讀方式](#成果檔案與解讀方式)
- [在自己的電腦執行](#在自己的電腦執行)
- [常見問題](#常見問題)
- [上傳 GitHub 與繳交](#上傳-github-與繳交)
- [限制與延伸實驗](#限制與延伸實驗)
- [參考資料](#參考資料)

## 學習目標

完成教學後，你應能：

1. 確認 Colab 是否取得 GPU，理解程式實際使用的裝置。
2. 分辨訓練集、驗證集、測試集的用途，避免資料洩漏。
3. 說明 CNN、損失函數、反向傳播與 Adam 的作用。
4. 執行訓練，依驗證 loss 保存最佳模型，再進行最終測試。
5. 解讀學習曲線、混淆矩陣與分類錯誤。
6. 保存實驗設定與成果，將可重現的專案整理到 GitHub。

建議先備知識：Python 變數、函式、迴圈與基本陣列概念。不需要自行準備 GPU。

## 專案內容

```text
README.md                         這份完整操作指南
notebooks/colab_gpu_training.ipynb 可獨立執行的 Colab 筆記本
src/training.py                   資料、模型、訓練及成果輸出
train.py                          命令列執行入口
requirements.txt                  本機執行所需套件
requirements-gpu.txt              本次 RTX 4060 實驗的固定 CUDA 套件版本
tests/                            訓練流程的重要檢查
docs/REPORT.md                    期中報告正文與實驗分析
docs/EXPERIMENTS.md               實驗紀錄與比較方法
results/README.md                 成果整理說明
results/local_cpu_demo/           本機 CPU 實際驗證紀錄
results/local_rtx4060/            本機 RTX 4060 GPU 實驗、圖表與訓練模型
outputs/                          你執行後產生的檔案（預設不納入 Git）
```

尚未建立 GitHub 儲存庫也能開始操作；筆記本不需要填入任何 GitHub 帳號或遠端網址。

## 從 Colab 開始

### 1. 上傳筆記本

1. 開啟 [Google Colab](https://colab.research.google.com/)。
2. 選擇「檔案 → 上傳筆記本」（File → Upload notebook）。
3. 選擇本專案的 [colab_gpu_training.ipynb](notebooks/colab_gpu_training.ipynb)。
4. 可選擇將副本儲存至自己的 Google Drive，方便保留筆記本。

Colab 可以載入 GitHub 或 Drive 中的筆記本，也可以直接上傳 `.ipynb`。筆記本內容與執行用的虛擬機器分開保存，因此保存筆記本不等於保存所有訓練檔案。[Colab 官方說明](https://research.google.com/colaboratory/intl/en-GB/faq.html)

### 2. 選擇 GPU 並檢查環境

1. 選擇「執行階段 → 變更執行階段類型」（Runtime → Change runtime type）。
2. 在硬體加速器選擇目前可用的 GPU，儲存設定。
3. 從筆記本第一格依序執行，或選擇「執行階段 → 全部執行」（Run all）。
4. 查看環境檢查輸出：CUDA 是否可用、實際裝置及 GPU 名稱。

本筆記本預設 `REQUIRE_GPU = True`，取得不到 GPU 時會停止並提供提示，避免誤把 CPU 執行當成 GPU 實驗。若要先以 CPU 練習，可明確改成 `REQUIRE_GPU = False`，並在報告標註使用 CPU。

**選到 GPU 執行階段只是第一步。** 程式還必須把模型與每批資料移到同一裝置；本專案的訓練流程已處理此步驟。[PyTorch 官方入門](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

Colab 的 GPU 供應、型號及使用額度會變動，不保證每次能取得同一硬體，也不保證固定可用時數。本教學不承諾特定 GPU 或完成時間。[Colab 資源限制](https://research.google.com/colaboratory/intl/en-GB/faq.html)

### 3. 執行訓練

第一次請保留筆記本「設定參數與保存位置」儲存格中的完整預設旗標；保留該格後續建立 `CONFIG` 的程式：

```python
REQUIRE_GPU = True
QUICK = False
EPOCHS = 5
```

完整模式使用 54,000 張訓練圖、6,000 張驗證圖與 10,000 張測試圖，共訓練 5 個 epochs。第一次執行需要下載 Fashion-MNIST；請等候下載完成。Colab 使用預裝的相容 PyTorch 與 torchvision，依筆記本的環境檢查操作即可，無須自行重裝 CUDA。

若只想先確認下載、訓練及檔案輸出能完成，可使用：

```python
QUICK = True
```

快速模式使用 2,048 張訓練圖、512 張驗證圖、512 張測試圖，且只訓練 1 個 epoch。**這是流程檢查，不能當作完整資料集的正式成果。** 確認後改回 `QUICK = False`，從參數格開始依序重新執行，包含資料載入、建立模型與訓練。修改旗標後，必須重新執行整個參數格，才會建立對應的 `CONFIG` 與新成果資料夾。

### 4. 查看結果與保存檔案

訓練後，筆記本會顯示學習曲線、混淆矩陣與預測範例，並產生模型與紀錄。每次實驗的預設輸出位於獨立的 `outputs/<時間戳記>/` 資料夾，避免混淆不同設定。

筆記本另提供兩種保存方式：

- **Google Drive**：依筆記本的選用步驟掛載 Drive，將輸出根目錄指向自己的實驗資料夾，再執行訓練。
- **下載 ZIP**：使用最後的選用下載儲存格，將本次輸出打包下載到電腦。

請在離開前確認模型、設定及圖表已保存。Colab 的虛擬機器會因閒置或生命週期限制被刪除，暫存磁碟中的檔案也可能隨之消失。[Colab 執行狀態與檔案保存](https://research.google.com/colaboratory/intl/en-GB/faq.html)

### 5. 載入模型與預測

依筆記本的「還原最佳模型」及「載入權重進行單張推論」段落，重新建立相同 CNN、載入 `best_model.pt`，再對測試影像產生預測。載入模型時須保留相同的前處理與類別順序。

訓練使用 `model.train()`；預測使用 `model.eval()` 與不計算梯度的範圍。這對含有 Dropout 的模型尤其重要。模型權重需要配合模型結構才能使用，不能只保存圖片或 accuracy 數字。[PyTorch 模型保存與載入](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## 資料與前處理

Fashion-MNIST 由 Zalando Research 提供，包含服飾商品的 28×28 灰階影像。官方資料有 60,000 張訓練圖及 10,000 張測試圖，共 10 類，適合示範影像分類流程。[資料集原始來源](https://github.com/zalandoresearch/fashion-mnist)

| 標籤 | 原始類別 | 中文說明 |
| --- | --- | --- |
| 0 | T-shirt/top | T 恤／上衣 |
| 1 | Trouser | 長褲 |
| 2 | Pullover | 套頭衫 |
| 3 | Dress | 洋裝 |
| 4 | Coat | 外套 |
| 5 | Sandal | 涼鞋 |
| 6 | Shirt | 襯衫 |
| 7 | Sneaker | 運動鞋 |
| 8 | Bag | 包包 |
| 9 | Ankle boot | 短靴 |

本專案的完整實驗切分如下：

| 資料 | 張數 | 用途 |
| --- | ---: | --- |
| 訓練集 | 54,000 | 計算梯度，更新模型參數 |
| 驗證集 | 6,000 | 每個 epoch 評估，選出驗證 loss 最低的模型 |
| 測試集 | 10,000 | 模型選定後，評估最終分類表現 |

訓練集與驗證集由官方 60,000 張訓練資料固定切分，seed 為 42；官方測試集不參與訓練與模型選擇。若反覆根據測試分數修改設定，測試資料的資訊就會影響模型選擇，使分數失去原本的獨立評估用途。[驗證與測試的分工](https://scikit-learn.org/stable/modules/cross_validation.html)

影像先轉成數值介於 0～1 的張量，再使用固定常數執行 `(x - 0.5) / 0.5`，將範圍轉成約 −1～1。這裡沒有利用驗證或測試資料估計平均值。未來若改成從資料計算正規化統計量，必須只使用訓練子集，再將同一轉換套用至其他集合。[資料洩漏與前處理](https://scikit-learn.org/stable/common_pitfalls.html)

## 神經網路如何學習

一次訓練更新包含以下步驟：

1. **取出一批資料**：DataLoader 將訓練集分成 mini-batches。
2. **前向傳播**：CNN 將影像轉成 10 個類別分數，稱為 logits。
3. **計算損失**：交叉熵比較 logits 與正確類別，量化目前的預測誤差。
4. **反向傳播**：計算損失對各參數的梯度。
5. **更新參數**：Adam 根據梯度調整權重；每次更新前清除上次梯度。

一個 **epoch** 代表模型完整看過當次訓練集一次。**batch size** 是每次更新使用的影像數；**learning rate** 控制更新步伐。步伐太大可能使損失不穩定，太小則可能學習緩慢。

本模型直接將 logits 交給 `CrossEntropyLoss`，損失計算前不額外加入 Softmax。推論時取分數最高的類別；需要顯示類別機率時，才將 logits 轉為 Softmax 機率。機率數值不保證等於實際正確率。[PyTorch 損失與最佳化教學](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

## CNN 模型架構

卷積層在影像不同位置使用相同的濾波器，學習局部圖樣；ReLU 引入非線性；最大池化縮小特徵圖。最後將特徵展平，交給全連接層進行 10 類分類。

`N` 代表當前批次的影像數；張量順序為「批次、通道、高、寬」。

| 階段 | 設定 | 輸出形狀 |
| --- | --- | --- |
| 輸入 | 灰階 28×28 | `N × 1 × 28 × 28` |
| 卷積 1 + ReLU | 1→16 通道，3×3，padding 1 | `N × 16 × 28 × 28` |
| 最大池化 1 | 2×2 | `N × 16 × 14 × 14` |
| 卷積 2 + ReLU | 16→32 通道，3×3，padding 1 | `N × 32 × 14 × 14` |
| 最大池化 2 | 2×2 | `N × 32 × 7 × 7` |
| 展平 | 32×7×7 | `N × 1568` |
| 全連接 + ReLU | 1568→128 | `N × 128` |
| Dropout | 機率 0.2，僅於訓練啟用 | `N × 128` |
| 輸出層 | 128→10 | `N × 10` logits |

模型共有 **206,922 個可訓練參數**。Dropout 在訓練時隨機遮蔽部分特徵，協助降低過度依賴特定特徵的情況；它不保證消除過擬合。

## 訓練與評估設定

| 項目 | 預設值 |
| --- | --- |
| epochs | 5 |
| batch size | 128 |
| optimizer | Adam |
| learning rate | 0.001 |
| loss | CrossEntropyLoss |
| seed | 42 |
| 驗證集張數 | 6,000 |
| DataLoader workers | 0 |
| 最佳模型標準 | 驗證 loss 最低 |

每個 epoch 先訓練，再評估驗證集。驗證時不更新權重，並以各批樣本數加權計算平均 loss。完成所有 epochs 後，載入最佳驗證模型，對測試集執行一次最終評估。

固定 seed 可減少隨機差異，但不同 PyTorch 版本、硬體與 CPU/GPU 執行仍可能產生不同結果。請連同環境資訊一起保存；不要將固定 seed 解讀為跨環境完全相同的保證。[PyTorch 重現性說明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

## 成果檔案與解讀方式

| 檔案 | 用途 |
| --- | --- |
| `config.json` | 本次訓練設定 |
| `environment.json` | 套件、裝置與執行環境資訊 |
| `history.csv` | 每個 epoch 的訓練與驗證紀錄 |
| `metrics.json` | 最終測試指標、最佳 epoch、切分張數等 |
| `best_model.pt` | 驗證 loss 最低時保存的模型 |
| `training_curves.png` | 訓練／驗證 loss 與 accuracy 曲線 |
| `confusion_matrix.png` | 真實類別與預測類別的對照 |
| `predictions.png` | 影像、真實標籤與預測標籤範例 |
| `misclassified.png` | 錯誤分類範例；沒有錯誤時不產生 |

以下是本機 RTX 4060 完整實驗的實際學習曲線：

![RTX 4060 完整訓練曲線](results/local_rtx4060/training_curves.png)

訓練 loss 從 0.5750 降到 0.2665，驗證 loss 從 0.3831 降到 0.2794；五輪中最低驗證 loss 位於第 5 輪，因此使用該輪權重進行最終測試。混淆矩陣顯示不同服飾類別仍有明顯差異：

![RTX 4060 測試混淆矩陣](results/local_rtx4060/confusion_matrix.png)

Shirt（襯衫）召回率為 70.60%，Sandal（涼鞋）為 98.40%。例如 136 張 T-shirt/top 被判為 Shirt，109 張 Pullover 被判為 Coat；詳細錯例與每類指標見 [實驗摘要](results/local_rtx4060/summary.md)。

`metrics.json` 主要欄位包括 `test_loss`、`test_accuracy`、`best_epoch`、`best_val_loss`、`train_seconds`、`parameter_count`、`split_sizes`、`quick_mode`、`per_class` 與 `macro_f1`。`test_accuracy` 為 0～1 的正確分類比例，報告成百分比時乘以 100。

`per_class` 記錄每類的 precision、recall、F1 與 support。precision 表示預測成該類的影像有多少正確；recall 表示真實屬於該類的影像有多少被找出；F1 綜合前兩者，Macro F1 則將 10 類的 F1 等權平均。這些指標能補充整體 accuracy 沒有呈現的類別差異。

解讀圖表時，請回答以下問題：

- **是否正在學習？** 訓練 loss 有沒有下降？驗證表現是否一起改善？
- **是否可能過擬合？** 若訓練持續改善、驗證 loss 卻上升，應討論泛化能力，並使用已保存的最佳驗證模型。
- **哪些類別較難？** 混淆矩陣的縱軸為真實類別、橫軸為預測類別；對角線表示分類正確，其他格子表示混淆。
- **錯誤是否有共同特徵？** 配合錯誤影像觀察輪廓、亮度或服飾外觀，區分可見現象與尚未驗證的推測。

本次數字、圖表與討論已整理於 [報告正文](docs/REPORT.md)。比較設定與紀錄方式請見 [實驗指南](docs/EXPERIMENTS.md)。

## 在自己的電腦執行

Colab 提供免自行準備 GPU 的教學路徑；本機也能使用相同訓練核心。本次 RTX 4060 Laptop GPU 實驗使用 Python 3.12.14、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128，CUDA runtime 為 12.8。建議使用 Python 3.12 與獨立的虛擬環境。

在具備相容 NVIDIA GPU 與驅動程式的本機，重現本次固定套件環境：

```bash
python -m pip install -r requirements-gpu.txt
python train.py --require-gpu --output-dir outputs/rtx4060_reproduction
```

`requirements-gpu.txt` 是本機環境紀錄，Colab 請保留預裝的相容套件。安裝完成後仍須確認 `torch.cuda.is_available()` 為 True，不能只憑電腦配有 GPU 判定程式已能使用 CUDA。

若只需要一般環境與 CPU 備援，可在專案資料夾開啟終端機，安裝一般依賴並執行：

```bash
python -m pip install -r requirements.txt
python train.py
```

一般命令會使用可用裝置；若你的目的明確是 GPU 實驗，可要求沒有 GPU 時停止：

```bash
python train.py --require-gpu
```

先做一次小規模流程檢查：

```bash
python train.py --quick
```

更改設定的範例：

```bash
python train.py --epochs 10 --batch-size 64 --learning-rate 0.0005
```

指定輸出位置：

```bash
python train.py --output-dir outputs/my_experiment
```

每次實驗請使用新的輸出資料夾，保存設定與結果的對應。檢查核心流程：

```bash
python -m unittest discover -s tests -v
```

上述測試用來檢查程式重要行為。實際 GPU 訓練結果須另外對照裝置、完整資料量及輸出檔案；本機 RTX 實驗也不代表 Colab 雲端環境已實測。

## 常見問題

| 問題 | 處理方式 |
| --- | --- |
| GPU 無法使用或 CUDA 為 False | 確認執行階段選擇 GPU，重新連線後重跑環境檢查。若 Colab 額度不足，稍後再試；想先練習可明確改用 CPU。 |
| 選了 GPU 仍看到 CPU | 查看實際裝置輸出；先重跑環境檢查及設定儲存格，再依順序執行。 |
| CUDA out of memory | 在參數格降低 `batch_size`，從整個參數格開始依序重跑，包含資料載入、模型建立與訓練，讓 DataLoader 使用新的 batch size。若仍無法執行，重新啟動執行階段後依序執行全部儲存格。 |
| 資料下載失敗 | 確認網路可用，稍後重新執行下載。不要把未完整下載的結果當成訓練失敗的證據。 |
| 出現變數尚未定義 | 筆記本可能跳格執行；重新啟動執行階段後，從第一格依序執行。 |
| 重連後模型或圖表消失 | 暫存環境可能已重建。從先前下載的 ZIP 或 Drive 取回成果；未保存時需要重新訓練。 |
| torch／torchvision 匯入錯誤 | Colab 先使用新的執行階段及預裝套件；本機則確認兩者相容，避免混用不同環境的安裝。 |
| 只有 quick 模式的數字 | 這只能證明子集流程可執行；請關閉 quick，重跑完整模式後再填正式成果。 |
| 每次準確率略有不同 | 先比對 seed、資料切分與環境紀錄；固定 seed 仍無法保證跨環境完全相同。 |

## 上傳 GitHub 與繳交

目前沒有預設 GitHub 儲存庫網址。請使用自己的帳號建立儲存庫；此專案不包含虛構的 Colab 徽章或遠端連結。

初學者可在 GitHub 網站建立新儲存庫，使用「Add file → Upload files」上傳專案檔案並保留資料夾結構。若你的網頁無法一次保留資料夾結構，可分別建立資料夾後上傳其中檔案。

上傳前檢查：

- [ ] README、筆記本、程式、依賴與測試檔案齊全。
- [ ] [REPORT.md](docs/REPORT.md) 的結果能對應實際執行環境與成果。
- [ ] GPU 實驗清楚標明本機或 Colab，並附實際 GPU 名稱與套件版本。
- [ ] 將正式成果複製至 `results/` 中獨立命名的實驗資料夾，附上執行設定與說明。
- [ ] 結果表能對應到同一輪的 `config.json`、`environment.json`、`history.csv` 與 `metrics.json`。
- [ ] 清楚標註本機 GPU、Colab、CPU 驗證及 quick 模式結果的身分。
- [ ] 不上傳 `.venv/`、資料集快取或無關私人檔案；本次小型 GPU 模型權重保留於 `results/local_rtx4060/best_model.pt`。
- [ ] 在 GitHub 頁面點開筆記本與文件，確認連結及圖片可讀。

## 限制與延伸實驗

Fashion-MNIST 的影像尺寸小、背景簡化；此模型的測試結果不能直接推論為真實拍照服飾辨識的效果。本教學的重點是建立完整、可檢查的訓練流程。

GPU 可平行處理大量運算，但小模型的資料載入、裝置傳輸與執行開銷也會影響時間，因此不能預先保證 GPU 一定比較快。若比較 CPU 與 GPU，需使用相同資料量、模型、epochs 與 batch size，明確界定計時範圍，並記錄硬體；CUDA 運算為非同步，計時需等待運算完成。[PyTorch CUDA 計時說明](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

可延伸比較 epochs、learning rate 或 batch size，每次只更改一項並使用相同驗證切分。先根據驗證結果選定設定，再評估測試集；不要為了提高測試分數反覆調參。完整紀錄方法見 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 參考資料

- [Google Colab 官方 FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)：GPU 與資源限制、執行環境、檔案保存。
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)：資料、模型、裝置與訓練流程。
- [PyTorch Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)：Fashion-MNIST 與批次資料載入。
- [PyTorch Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：損失函數、梯度與參數更新。
- [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：模型保存與推論模式。
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)：隨機性與重現限制。
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)：GPU 非同步運算與時間測量。
- [Zalando Research：Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)：資料集定義與類別。
- [scikit-learn：Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)：訓練、驗證與測試的分工。
- [scikit-learn：Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：前處理一致性及資料洩漏。
