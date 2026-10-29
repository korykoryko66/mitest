# 指定文章版本驗證紀錄

日期：2026-10-29（Asia/Taipei）。本次依[指定文章](https://zhuanlan.zhihu.com/p/379291795)重新實作；原文設定、必要修正及本報告補強見 [SOURCE_ALIGNMENT.md](SOURCE_ALIGNMENT.md)。

## 已完成

- **核心測試：11 項通過。** 包含固定且不重疊的資料切分、非法設定拒絕、quick 固定一輪、MLP 更新權重與 25,450 個參數、巢狀資料及模型的裝置搬移、DeviceDataLoader、CUDA 訓練、兩階段 SGD 學習率、尾端小批次的加權指標、混淆矩陣方向及最佳 checkpoint 還原。
- **最佳模型選擇有刻意退步的測試。** 測試使驗證 loss 在後續輪次惡化，要求還原較早的最低驗證 loss 權重，避免只驗證最後一輪恰好最佳的情況。
- **原文設定已核對。** MNIST、僅 ToTensor、50,000／10,000 訓練驗證切分、784→32→10 MLP、train batch 128、validation batch 256；依作者 notebook 先 5 輪 SGD lr 0.5，再以同一模型續訓 5 輪 lr 0.1。固定 seed、獨立測試、樣本加權及模型保存等補強均有標示。
- **Notebook 來源與格式通過。** 39 個儲存格、22 個程式格可編譯，依原文 11 步編排，且與產生器及共用來源一致；另以 nbformat 驗證官方格式。
- **Notebook 獨立執行通過。** 在沒有 src/ 的獨立資料夾依序執行全部程式格，驗證時明確強制 CPU、quick 及單輪。流程完成訓練、最佳模型還原、測試、製圖及單張推論。這次 quick 成績僅用於確認流程，不列為正式成果。提交的筆記本仍預設完整資料、10 輪與 GPU 必要檢查。
- **RTX 4060 完整 GPU 實驗通過。** 訓練／驗證／測試為 50,000／10,000／10,000 張；測試 accuracy **96.99%（9,699／10,000）**、test loss 0.100092，最佳驗證 epoch 10、驗證 loss 0.114256。訓練與驗證計時 **43.48 秒**；包含首次資料下載、初始驗證、訓練、最終測試與製圖的完整呼叫為 **65.93 秒**。計時範圍詳見[實驗摘要](../results/mnist_rtx4060/summary.md)。
- **環境與硬體證據已保存。** NVIDIA GeForce RTX 4060 Laptop GPU、8188 MiB、驅動 610.74、Python 3.12.14、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、CUDA runtime 12.8。實際裝置為 cuda，並通過 CUDA 張量運算。請見[環境](../results/mnist_rtx4060/environment.json)、[硬體](../results/mnist_rtx4060/gpu_hardware.json)及[執行紀錄](../results/mnist_rtx4060/gpu_runtime.json)。
- **真實權重的跨裝置載入通過。** 已訓練的 best_model.pt 能在 CPU 與 GPU 載入並推論；兩張合成輸入的 logits 最大絕對差為 1.91×10⁻⁶，容許誤差檢查通過。未為此重新評估測試集。checkpoint 的 epoch、驗證 loss、設定、10 輪 history、混淆矩陣總數及來源 SHA-256 已核對，請見[權重驗證](../results/mnist_rtx4060/checkpoint_verification.json)。
- **四張成果圖已目視檢查。** 學習曲線、混淆矩陣、預測與錯例圖的標籤清楚，混淆矩陣方向及數值與 CSV 一致。

## 重做檢查

先依 README 安裝相依套件，再於專案根目錄執行：

```bash
python -m unittest discover -s tests -v
python scripts/verify_notebook.py
python scripts/verify_notebook.py --execute --data-dir data
```

第三項會明確強制 CPU quick 模式；預設在暫存資料夾執行，結束後清除該暫存輸出。若要保留驗證成果，可加上 `--work-dir outputs/notebook_check`，並使用沒有 src/ 的獨立位置。

修改 src/training.py 後，先執行 `python scripts/build_notebook.py` 重新產生 notebook，再做來源一致性檢查。原始正式成果保留於 [metrics.json](../results/mnist_rtx4060/metrics.json)、[history.csv](../results/mnist_rtx4060/history.csv)與[混淆矩陣數值](../results/mnist_rtx4060/confusion_matrix.csv)。

## 執行環境與限制

正式實驗是在這台本機 RTX 4060 筆電實測，並未在使用者的 Colab 帳號上實際執行；Colab 雲端、Drive 掛載與瀏覽器下載仍需使用者在自己的帳號操作。筆記本保留相同核心與完整教學，不將本機成果冒稱為 Colab 雲端成果。

`REQUIRE_GPU = False` 代表允許 CPU 備援；若 CUDA 可用仍優先使用 GPU，並不代表強制 CPU。本機 Windows 的 workers=0、驗證不打亂及加權彙總等與原文差異，均已註明。

本次僅單一 seed、單次正式訓練；未提供多次變異、受控的 CPU／GPU 速度比較或不同 GPU 的加速倍數。完整模型載入與計時限制請見 [EXPERIMENTS.md](EXPERIMENTS.md)。
