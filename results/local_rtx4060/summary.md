# 本機 RTX 4060 Laptop GPU 完整實驗成果

2026-10-29 約 08:38（Asia/Taipei），在這台 Windows 筆電上，以 **NVIDIA GeForce RTX 4060 Laptop GPU** 完成 Fashion-MNIST 完整資料的 5 輪 CNN 訓練。測試集 10,000 張中有 8,940 張分類正確，**accuracy 為 89.40%**。這是本機 CUDA 實測，並非 Colab 雲端硬體的測量。

## 環境與設定

| 項目 | 實際內容 |
| --- | --- |
| 系統／GPU | Windows 11／NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU 記憶體／驅動 | 硬體回報 8,188 MiB／610.74 |
| 裝置／CUDA 張量運算 | cuda／已驗證成功 |
| Python | 3.12.14 |
| PyTorch／torchvision | 2.9.1+cu128／0.24.1+cu128 |
| CUDA runtime／NumPy | 12.8／2.5.4 |
| PyTorch CPU 運算執行緒 | 4 |
| 訓練／驗證／測試張數 | 54,000／6,000／10,000 |
| epochs／batch size／learning rate | 5／128／0.001 |
| seed／quick | 42／false |
| optimizer／loss | Adam／CrossEntropyLoss |
| 最佳模型標準 | 最低驗證 loss |

本次實際由 `src.training.run_experiment` 函式執行，另保存硬體及 PyTorch 顯示記憶體峰值。下列命令可重現相同訓練核心，並使用新的輸出位置：

```bash
python -m pip install -r requirements-gpu.txt
python train.py --require-gpu --output-dir outputs/rtx4060_repeat
```

固定套件版本適用本次 Python 3.12 的本機 CUDA 環境；Colab 使用教學筆記本中的預裝相容套件。上述命令為等價重現方式，實際函式入口與執行時間另記錄於 [gpu_runtime.json](gpu_runtime.json)。

## 實際結果

| 指標 | 數值 |
| --- | ---: |
| 最佳驗證 epoch | 5 |
| 最低驗證 loss | 0.279371 |
| 最佳 epoch 的驗證 accuracy | 89.57% |
| 最終測試 loss | 0.289209 |
| 最終測試 accuracy | **89.40%** |
| 正確／錯誤張數 | 8,940／1,060 |
| Macro F1 | 0.894313 |
| 訓練與驗證計時 | 48.91 秒 |
| 外層完整實驗呼叫時間 | 51.63 秒 |
| 可訓練參數 | 206,922 |
| PyTorch 最高張量配置量 | 43,897,856 bytes，約 41.86 MiB |
| PyTorch 最高保留記憶體 | 71,303,168 bytes，68.00 MiB |

48.91 秒包含 `fit` 內每輪的訓練、驗證、最佳權重與歷史紀錄寫入，不包含最終測試及圖表產生。51.63 秒為外層整個實驗函式呼叫，兩者範圍不同；本次資料已快取，套件安裝與首次下載另計。PyTorch 記憶體統計不等於整張 GPU 占用；本報告沒有宣稱 GPU 利用率或功耗。

## 學習曲線

![RTX 4060 完整訓練的學習曲線](training_curves.png)

訓練 loss 由 0.574962 降至 0.266514，驗證 loss 由 0.383078 降至 0.279371；驗證 accuracy 從 85.93% 提高到 89.57%。五輪中最低驗證 loss 出現在第 5 輪，因此選用該輪模型進行最終測試。原始逐輪資料見 [history.csv](history.csv)。觀察範圍內 loss 持續改善，但不能推論增加更多 epochs 一定繼續改善。

## 各類別與錯誤分析

![RTX 4060 測試混淆矩陣](confusion_matrix.png)

每個測試類別各有 1,000 張。Shirt 的 precision 為 69.97%、recall 為 70.60%、F1 為 0.7028，是本次較弱的類別；Sandal 的 recall 為 98.40%，Trouser 的 precision 為 99.49%。完整每類指標見 [metrics.json](metrics.json)。

矩陣縱軸為真實類別、橫軸為預測類別。明顯混淆包括：

| 真實類別 → 預測類別 | 張數 |
| --- | ---: |
| T-shirt/top → Shirt | 136 |
| Pullover → Coat | 109 |
| Shirt → Coat | 100 |
| Shirt → T-shirt/top | 99 |
| Pullover → Shirt | 76 |

![RTX 4060 模型預測範例](predictions.png)

![RTX 4060 模型錯誤分類範例](misclassified.png)

錯例可見 Coat／Pullover、Dress／Shirt 與 Ankle boot／Sneaker 的混淆。部分輪廓相近、細節有限，可作後續分析線索；影像觀察本身尚不是模型判錯原因的因果證據。

## 已訓練模型與載入檢查

[best_model.pt](best_model.pt) 為本次最佳驗證 checkpoint，大小 831,373 bytes，約 0.83 MB，已保留供下載與推論。它包含模型權重及相應 epoch、驗證 loss、設定；載入時使用相同 CNN 與前處理。

[checkpoint_verification.json](checkpoint_verification.json) 確認完整資料設定、5 輪紀錄、最佳驗證 checkpoint 及檔案雜湊相符。本次 GPU 權重已分別在 CPU 與 GPU 載入，對 `2 × 1 × 28 × 28` 的合成影像推論，logits 最大絕對差約為 `7.15 × 10⁻⁷`。這是載入與推論檢查，沒有再次評估測試集，也不是新的準確率實驗。

## 先前 CPU 紀錄的定位

[先前 CPU 完整紀錄](../local_cpu_demo/summary.md) 的測試 accuracy 為 89.57%，訓練與驗證計時 51.97 秒；本次 GPU 為 89.40% 與 48.91 秒。主要訓練設定相同，但 CPU 使用 `+cpu` 套件，GPU 使用 `+cu128` 套件，各只有單次觀察。因此不將時間差宣稱為純硬體加速倍數，也不以 0.17 個百分點的分數差判定裝置優劣。

## 原始證據

- [config.json](config.json)：完整訓練設定。
- [environment.json](environment.json)：實際軟體、裝置與時間。
- [gpu_hardware.json](gpu_hardware.json)：GPU、記憶體與驅動。
- [gpu_runtime.json](gpu_runtime.json)：CUDA 確認、計時、記憶體及雜湊。
- [history.csv](history.csv)：逐輪原始紀錄。
- [metrics.json](metrics.json)：最終測試及各類指標。
- [training.log](training.log)：實際執行輸出。
- [checkpoint_verification.json](checkpoint_verification.json)：已訓練模型載入檢查。

完整方法與結論見 [期中報告](../../docs/REPORT.md)，重做實驗的紀錄原則見 [實驗指南](../../docs/EXPERIMENTS.md)。
