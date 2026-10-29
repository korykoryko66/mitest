# 驗證紀錄

日期：2026-10-29（Asia/Taipei）。

## 已完成

- **核心測試：7 項通過。** 包含切分完整、不重疊且可重現，非法設定拒絕、quick 固定一輪、CNN 真正更新權重、尾端小 batch 的 loss 加權、混淆矩陣方向，以及最佳 checkpoint 的保存與 CPU 載入。
- **最佳模型測試刻意製造驗證 loss 惡化。** 最佳輪次早於最後一輪，載入的權重須對應最低驗證 loss，避免僅測到「剛好最後一輪最好」的情況。
- **Notebook 來源與格式通過。** 28 個儲存格、15 個程式格可編譯，符合官方 nbformat schema，且與產生器及共用程式來源一致。
- **Notebook 獨立執行通過。** 在沒有 src/ 的工作資料夾順序執行所有程式格；僅為驗證明確改為 CPU、quick 與單一 epoch。流程完成訓練、最佳權重還原、評估、圖表與單張推論。提交的筆記本仍預設完整資料與 GPU 必要檢查。
- **CPU 完整實驗通過。** 54,000／6,000／10,000 張資料，5 個 epochs，測試 accuracy 89.57%。原始環境、設定、每輪紀錄與圖表見 [實驗摘要](../results/local_cpu_demo/summary.md)。
- **RTX 4060 完整 GPU 實驗通過。** 同樣使用完整切分與 5 個 epochs，在這台筆電實際以 CUDA 訓練，最終測試 accuracy 89.40%、test loss 0.289209、最佳驗證 epoch 為 5。訓練與驗證計時 48.91 秒；含最終評估與製圖的整輪流程為 51.63 秒。完整數字以 [GPU 原始指標](../results/local_rtx4060/metrics.json) 為準。
- **GPU 硬體與環境已保存。** NVIDIA GeForce RTX 4060 Laptop GPU、8 GB、驅動 610.74、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、CUDA runtime 12.8；`device=cuda` 且 `cuda_available=true`。請見 [環境](../results/local_rtx4060/environment.json)、[硬體](../results/local_rtx4060/gpu_hardware.json) 與 [執行紀錄](../results/local_rtx4060/gpu_runtime.json)。
- **真實訓練權重的跨裝置載入通過。** GPU 實驗的 best_model.pt 能在 CPU 與 GPU 載入、執行推論；用兩張合成輸入比對，logits 最大絕對差為 7.15×10⁻⁷。驗證未重新計算測試集分數。另已核對 checkpoint 的 epoch／驗證 loss／設定、五輪 history 與來源 SHA-256，結果見 [權重驗證](../results/local_rtx4060/checkpoint_verification.json)。
- **GPU 套件依賴檢查通過。** `python -m pip check` 回報沒有衝突；更換 GPU 套件後，核心 7 項測試與 notebook 來源一致性已重新通過。
- **文件交叉檢查完成。** 已修正設定旗標與 GPU 必要檢查的對應，並修正記憶體不足後需從參數格重建 DataLoader 的步驟。
- 已目視檢查 CPU 與 RTX 4060 的學習曲線、混淆矩陣；圖表標籤、方向與數字可讀。

## 重做檢查

先依 README 安裝相依套件，再於專案根目錄執行：

```bash
python -m unittest discover -s tests -v
python scripts/verify_notebook.py
python scripts/verify_notebook.py --execute --data-dir data
```

第三項會明確使用 CPU quick 模式；預設在暫存資料夾中執行，結束後清除該暫存輸出。若要保留驗證成果，可加上 `--work-dir outputs/notebook_check`，並使用沒有 src/ 的獨立位置。

修改 `src/training.py` 後，先執行 `python scripts/build_notebook.py` 重新產生 notebook，再做來源一致性檢查。

## 執行環境與限制

本報告的主要實驗已在這台 RTX 4060 筆電完成。先前環境安裝的是 CPU 版 PyTorch；本次已在專案的隔離環境改用 CUDA 版，並確認真實 CUDA 張量運算。可重現本機 GPU 環境的相依檔案為 [requirements-gpu.txt](../requirements-gpu.txt)。

Colab 雲端、Drive 掛載與瀏覽器下載未在使用者帳號實測。筆記本提供相同訓練核心與完整操作步驟，預設 GPU 必要檢查；本機 RTX 成果不冒稱為 Colab 雲端成果。

CPU 紀錄保留作次要參考；兩次測量使用不同的 PyTorch 組建後綴，未以此宣稱受控的 GPU 加速倍數。模型僅單次固定 seed 訓練，跨硬體、版本與多次執行的變異仍是限制。
