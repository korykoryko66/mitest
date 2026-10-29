# MNIST 實驗設定與證據解讀

本文件說明「**完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型**」的實驗規則。設定以[指定文章](https://zhuanlan.zhihu.com/p/379291795)與[作者 notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)為依據；新增的測試、保存與重現措施見 [SOURCE_ALIGNMENT.md](SOURCE_ALIGNMENT.md)。

## 正式實驗與快速檢查

| 項目 | 完整實驗 | quick 流程檢查 |
| --- | --- | --- |
| 資料 | MNIST，僅 ToTensor | 同一資料與前處理 |
| 訓練／驗證／測試 | 50,000／10,000／10,000 | 2,048／512／512 |
| epochs | 10 | 1 |
| 學習率 | 前 5 輪 0.5，後 5 輪 0.1 | 第 1 輪 0.5 |
| 模型／optimizer | MLP 784→32→10／SGD | 同一模型與 optimizer |
| 用途 | 完整分類成果 | 確認流程可執行，不能代替正式成果 |

共同設定為 train batch 128、validation／test batch 256、seed 42、hidden size 32。Windows 使用 workers 0；若在 Colab 改為 2，應保存該輪設定。

## 兩階段學習率

作者 notebook 分別呼叫兩次 `fit`：先 5 輪 0.5，再 5 輪 0.1，同一模型繼續更新。本專案在第 6 輪切換到 0.1，重建無 momentum SGD，並將實際 learning rate 寫入 `history.csv`。`switch_epoch=5` 表示前 5 輪使用第一階段，第 6 輪起進入第二階段。

不以測試分數決定切換時機。模型選擇依驗證 loss；完成訓練後載入最佳權重，對測試集評估一次。訓練前的初始驗證記錄在獨立檔案，不列成第 1 輪。

## 本機與 Colab

本次 MNIST 完整 GPU 實驗已完成：測試 accuracy 96.99%、Macro F1 0.969714，最佳驗證 epoch 為 10。訓練與驗證計時為 43.48 秒；包含資料下載、初始驗證、測試及圖表的外層完整呼叫為 65.93 秒。[實際摘要](../results/mnist_rtx4060/summary.md)

本機完整 GPU 執行使用：

```bash
python train.py --require-gpu --output-dir outputs/mnist_repeat
```

本次實際透過相同 `run_experiment` 核心函式執行，以上為等價重現命令。GPU 成果保存於 `results/mnist_rtx4060/`。若在 Colab 重做，另建實驗資料夾，記錄當次 GPU 與版本；不要把本機 RTX 4060 的數字標為 Colab 雲端結果。

Colab 操作為上傳筆記本、選 GPU、確認完整模式、從頭依序執行、檢查圖表、使用 Drive 或 ZIP 保存。修改參數後需重跑整個參數格及後續資料、模型與訓練步驟。[Colab 官方 FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)

## 證據檔案

| 要確認的事項 | 對應檔案 |
| --- | --- |
| 模型、epochs、batch、seed 與學習率設定 | `config.json` |
| Python、PyTorch、CUDA 與實際裝置 | `environment.json` |
| 未訓練模型的起點 | `initial_validation.json` |
| 每輪 loss、accuracy、learning rate 與時間 | `history.csv` |
| 最佳驗證 epoch 與最終測試 | `metrics.json` |
| 數字分類錯誤分布 | `confusion_matrix.csv`、`confusion_matrix.png` |
| 實際錯例 | `misclassified.png` |
| 可還原模型 | `best_model.pt`、`checkpoint_verification.json` |
| GPU 硬體及執行資訊 | `gpu_hardware.json`、`gpu_runtime.json` |

accuracy、precision、recall 在原始紀錄中是比例，呈現百分比需乘以 100。Macro F1 是 10 個數字類別 F1 的等權平均。不要把某個 epoch 的驗證 accuracy 寫成最終測試 accuracy。

## 解讀結果

**學習曲線**：比較訓練與驗證 loss，並觀察第 6 輪切換學習率前後的變化。單次曲線只能說明該次觀察，不能僅因階段切換與改善同時出現就證明因果。

**混淆矩陣**：列／縱軸為真實數字、欄／橫軸為預測數字。以 CSV 確認具體混淆數量，再配合錯例觀察筆畫、斜度或影像模糊等可見現象；原因分析須區分觀察與推測。

**初始與訓練後**：初始驗證提供尚未學習的參考點，訓練後比較使用驗證資料，最終測試則維持獨立。不同資料集合的指標不要混為同一測量。

**計時**：`train_seconds` 依程式的 fit 範圍解讀；外層整個實驗時間可能另外包含資料準備、測試與畫圖，兩者需分列。CUDA 為非同步運算，準確計時應在邊界等待 GPU 完成。[PyTorch CUDA 說明](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

## 重做與比較

每輪使用新的輸出資料夾；程式會拒絕覆寫已有成果。固定 seed 42 有助降低切分與初始化差異，仍不能保證跨硬體與套件版本逐位一致。[PyTorch 重現性說明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

若延伸比較隱藏層大小、學習率或批次，每次只改一項，以同一驗證切分比較，保留所有嘗試。若要比較 CPU／GPU 速度，需另統一軟體環境、工作量、計時範圍並重複測量；本報告不由單次執行推論所有 GPU 的加速幅度。

保存筆記本不等於保存執行環境中的模型與圖表。請在 Colab 斷線前確認 Drive 或 ZIP 包含設定、環境、紀錄、指標、模型及圖表。
