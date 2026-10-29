# MNIST：本機 RTX 4060 Laptop GPU 實驗成果

本次依[指定知乎文章](https://zhuanlan.zhihu.com/p/379291795)及[作者 notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)的主線，以 MNIST、單隱藏層 MLP 和兩階段 SGD 完成 **10 輪實際 GPU 訓練**。獨立測試 10,000 張中有 9,699 張正確，accuracy 為 **96.99%**。

執行位置為本機 Windows 筆電，使用 NVIDIA GeForce RTX 4060 Laptop GPU；這些數字是本機實測，不是 Colab 雲端硬體的測量。完整方法與來源修正見 [期中報告](../../docs/REPORT.md) 及 [來源對照](../../docs/SOURCE_ALIGNMENT.md)。

## 環境與設定

| 項目 | 實際內容 |
| --- | --- |
| 日期／時間 | 2026-10-29，13:35～13:36，Asia/Taipei |
| 系統／GPU | Windows 11／NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU 記憶體／驅動 | 硬體回報 8,188 MiB／610.74 |
| 裝置／CUDA 張量運算 | cuda／已驗證成功 |
| Python | 3.12.14 |
| PyTorch／torchvision | 2.9.1+cu128／0.24.1+cu128 |
| CUDA runtime／NumPy | 12.8／2.5.4 |
| CPU 運算執行緒／DataLoader workers | 4／0 |
| 資料／前處理 | MNIST／僅 ToTensor，沒有額外 Normalize |
| 訓練／驗證／測試張數 | 50,000／10,000／10,000 |
| 模型／參數量 | MLP 784→32→10、ReLU／25,450 |
| 訓練／驗證及測試 batch size | 128／256 |
| optimizer／loss | SGD，無 momentum／CrossEntropyLoss |
| epochs／learning rate | 10／第 1～5 輪 0.5，第 6～10 輪 0.1 |
| seed／quick | 42／false |
| 最佳模型標準 | 最低驗證 loss |

本次實際由 `src.training.run_experiment` 函式完成，外層另保存 GPU 硬體、計時與 PyTorch 記憶體峰值。等價重現命令如下，請使用新的輸出位置：

```bash
python -m pip install -r requirements-gpu.txt
python train.py --require-gpu --output-dir outputs/mnist_reproduction
```

套件固定版本對應本次 Python 3.12 的本機 CUDA 環境。Colab 使用筆記本中的預裝相容套件；本機命令不應原封不動作為 Colab 套件重裝要求。

## 初始驗證與最終結果

| 指標 | 數值 |
| --- | ---: |
| 未訓練模型驗證 loss | 2.310835 |
| 未訓練模型驗證 accuracy | 9.86% |
| 最佳驗證 epoch | 10 |
| 最低驗證 loss | 0.114256 |
| 最佳 epoch 的驗證 accuracy | 96.57% |
| 最終測試 loss | 0.100092 |
| 最終測試 accuracy | **96.99%** |
| 正確／錯誤張數 | 9,699／301 |
| Macro F1 | 0.969714 |
| 訓練與驗證計時 | 43.48 秒 |
| 外層完整實驗呼叫時間 | 65.93 秒 |

初始驗證記錄於 [initial_validation.json](initial_validation.json)，最終測試於 [metrics.json](metrics.json)。驗證與測試是不同集合，不能把初始驗證和最終測試當成同一批資料上的前後測。

43.48 秒包含 fit 中的訓練、驗證、checkpoint 與歷史紀錄寫入，不包含最終測試及圖表。65.93 秒包含本次資料下載、初始驗證、訓練、最終測試及圖表，不包含套件安裝；兩個計時範圍不同。

PyTorch 最高張量配置量為 20,016,640 bytes（約 19.09 MiB），最高保留量為 25,165,824 bytes（24.00 MiB）。這些配置器統計不等於整張 GPU 占用，也不是利用率或功耗。[GPU 執行紀錄](gpu_runtime.json)

## 兩階段學習紀錄

| Epoch | Learning rate | 訓練 loss | 訓練 accuracy | 驗證 loss | 驗證 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 0.402889 | 87.91% | 0.241746 | 92.50% |
| 2 | 0.5 | 0.195261 | 94.22% | 0.201514 | 93.80% |
| 3 | 0.5 | 0.150845 | 95.56% | 0.214684 | 93.41% |
| 4 | 0.5 | 0.126522 | 96.27% | 0.141794 | 95.60% |
| 5 | 0.5 | 0.109304 | 96.75% | 0.156530 | 95.15% |
| 6 | 0.1 | 0.082494 | 97.58% | 0.117705 | 96.38% |
| 7 | 0.1 | 0.077386 | 97.76% | 0.118750 | 96.36% |
| 8 | 0.1 | 0.074832 | 97.80% | 0.116957 | 96.38% |
| 9 | 0.1 | 0.072635 | 97.86% | 0.115005 | 96.42% |
| 10 | 0.1 | 0.070519 | 98.02% | 0.114256 | 96.57% |

![MNIST RTX 4060 的實際學習曲線](training_curves.png)

驗證 loss 在第 3、5、7 輪略有反彈，最低值位於第 10 輪。第 6 輪學習率切換後驗證 loss 下降，但這不是固定學習率的受控比較，不能將改善全部歸因於切換。每輪原始資料見 [history.csv](history.csv)。

## 分類錯誤與影像

![MNIST 測試混淆矩陣](confusion_matrix.png)

混淆矩陣的列／縱軸為真實數字，欄／橫軸為預測數字。每個數字測試張數不同，精確樣本數及 precision、recall、F1 在 [metrics.json](metrics.json)。數字 9 的 recall 為 95.24%，本次最低；數字 0 為 99.08%。

| 真實數字 → 預測數字 | 張數 |
| --- | ---: |
| 5 → 3 | 16 |
| 4 → 9 | 14 |
| 9 → 4 | 13 |
| 7 → 2 | 12 |
| 7 → 3 | 10 |

上述數量可由 [confusion_matrix.csv](confusion_matrix.csv) 直接核對。

![MNIST 模型預測範例](predictions.png)

![MNIST 模型實際錯誤範例](misclassified.png)

錯例可見 5／6、9／8、6／0、2／7 的混淆。筆畫的封閉環或相似輪廓提供觀察線索，但不足以單獨證明模型判斷的原因。整體還有 301 張測試錯誤，96.99% 不能直接推論到真實拍照或不同字體資料。

## 已訓練模型與載入檢查

[best_model.pt](best_model.pt) 是第 10 輪最佳驗證 checkpoint，大小 104,413 bytes，約 0.10 MB，包含權重與對應訓練資訊。還原時使用相同 MLP 結構與 ToTensor 前處理，推論切至評估模式。

[checkpoint_verification.json](checkpoint_verification.json) 確認 MNIST 完整資料與設定、10 輪 SGD 學習率、最佳驗證權重、來源與權重雜湊，以及混淆矩陣對應 9,699／10,000 張正確。已訓練權重分別在 CPU 與 GPU 載入，對 `2 × 1 × 28 × 28` 合成影像推論，logits 最大絕對差約 `1.91 × 10⁻⁶`。這是載入檢查，沒有再次評估測試集。

## 原始證據

- [config.json](config.json)：模型、資料切分、seed 及兩階段設定。
- [environment.json](environment.json)：軟體版本、裝置與時間。
- [initial_validation.json](initial_validation.json)：未訓練模型的驗證起點。
- [history.csv](history.csv)：10 輪原始紀錄與 learning rate。
- [metrics.json](metrics.json)：獨立測試及每類指標。
- [confusion_matrix.csv](confusion_matrix.csv)：可核對的分類計數。
- [gpu_hardware.json](gpu_hardware.json)：GPU、記憶體與驅動。
- [gpu_runtime.json](gpu_runtime.json)：執行位置、計時、CUDA 與記憶體證據。
- [checkpoint_verification.json](checkpoint_verification.json)：模型與成果一致性檢查。
- [training.log](training.log)：實際執行輸出。

重做與比較原則見 [EXPERIMENTS.md](../../docs/EXPERIMENTS.md)。本次只有單次 GPU 完整實驗，不宣稱 CPU／GPU 加速倍數；固定 seed 也不保證跨環境逐位相同。
