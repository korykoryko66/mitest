# MNIST 實驗成果

本資料夾整理「**完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型**」的 MNIST 實驗證據。方法依[指定文章](https://zhuanlan.zhihu.com/p/379291795)，使用 784→32→10 MLP 與兩階段 SGD；來源與本報告補強見 [SOURCE_ALIGNMENT.md](../docs/SOURCE_ALIGNMENT.md)。

本次 RTX 4060 完整 MNIST 實驗已完成，成果保存於 [mnist_rtx4060/summary.md](mnist_rtx4060/summary.md)。數字來自該資料夾的本次輸出，未使用原作者的驗證分數代替。

| 指標 | 本次實測 |
| --- | ---: |
| 訓練／驗證／測試張數 | 50,000／10,000／10,000 |
| 最佳驗證 epoch | 10 |
| 最終測試 accuracy | **96.99%（9,699／10,000）** |
| 測試 loss／Macro F1 | 0.100092／0.969714 |
| 訓練與驗證計時 | 43.48 秒 |

![MNIST RTX 4060 實際學習曲線](mnist_rtx4060/training_curves.png)

本次已訓練的 [best_model.pt](mnist_rtx4060/best_model.pt) 約 0.10 MB，可配合相同模型結構及前處理載入。完整環境、兩階段學習紀錄、錯例與限制見 [實驗摘要](mnist_rtx4060/summary.md) 與 [報告](../docs/REPORT.md)。

## 證據內容

| 檔案 | 內容 |
| --- | --- |
| `summary.md` | 實驗環境、成果、分析與限制 |
| `config.json`、`environment.json` | 設定與實際裝置 |
| `initial_validation.json` | 未訓練模型驗證 |
| `history.csv` | 10 輪 loss、accuracy 與 learning rate |
| `metrics.json` | 最佳驗證模型的最終測試及每類指標 |
| `best_model.pt` | 已訓練的最佳 checkpoint |
| `training_curves.png` | 訓練與驗證曲線 |
| `confusion_matrix.csv`、`confusion_matrix.png` | 分類錯誤統計 |
| `predictions.png`、`misclassified.png` | 預測與實際錯例 |
| `gpu_hardware.json`、`gpu_runtime.json` | GPU 執行與硬體證據 |
| `checkpoint_verification.json` | 權重一致性與載入檢查 |

本機 GPU 與 Colab 雲端分開標示。快速模式 2,048／512／512 張、1 輪只檢查流程，不能當成本次完整 50,000／10,000／10,000 張的成果。再次實驗須使用新資料夾，並保留設定、環境與結果的對應。[實驗指南](../docs/EXPERIMENTS.md)
