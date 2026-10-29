# 實驗紀錄與比較指南

本文件配合「**完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型**」，說明本專案的實驗紀錄方式及比較原則。結果應能追溯到同一輪實驗的設定、環境與成果檔案。本機 GPU 執行與 Colab 雲端 GPU 執行分別標示。

## 分清楚實驗與執行環境

| 執行 | 資料與訓練量 | 能支持的結論 |
| --- | --- | --- |
| quick 流程檢查 | 2,048 訓練／512 驗證／512 測試，1 epoch | 這組小規模流程能否完成；不能代表完整資料成果 |
| 本機 RTX 4060 Laptop GPU | 54,000 訓練／6,000 驗證／10,000 測試，5 epochs | 本報告主要實驗，結果以 [GPU 摘要](../results/local_rtx4060/summary.md) 為準 |
| 本機 CPU 基準紀錄 | 相同資料量與主要訓練設定，5 epochs | 先前 CPU 軟體環境的單次觀察，屬次要歷史紀錄 |
| Colab 雲端 GPU | 教學筆記本提供相同訓練核心與完整模式 | 雲端環境尚未實測，不能把本機 RTX 結果當成 Colab 硬體成果 |

完整模式使用的模型與預設參數詳見 [README](../README.md)。本機 GPU 與 Colab 共用資料、模型及訓練邏輯，但硬體、套件與檔案環境不同，執行位置必須分別標示。

## 本機 GPU 實驗設定

本報告使用 RTX 4060 Laptop GPU，固定 seed 42、batch size 128、learning rate 0.001、5 epochs、Adam 與交叉熵；以驗證 loss 選擇最佳模型，再對官方 10,000 張測試圖評估一次。完整執行使用：

```bash
python train.py --require-gpu --output-dir results/local_rtx4060
```

成果資料夾已存在時，程式會拒絕覆寫。再次實驗需指定新的輸出位置，讓設定與結果保持一對一對應。

## 在 Colab 重做相同流程

1. 開啟筆記本，選擇 GPU 執行階段，確認輸出的實際 GPU 名稱。
2. 在筆記本參數格確認 `REQUIRE_GPU = True`、`QUICK = False`、`EPOCHS = 5`，保留後續建立 `CONFIG` 的程式，並核對 batch size、learning rate 與 seed。修改後從整個參數格開始依序重跑，包含資料載入與模型建立。
3. 決定成果保存位置，從頭依序執行筆記本。
4. 確認訓練與驗證每個 epoch 均有紀錄，並已保存最佳驗證模型。
5. 確認最終測試使用最佳模型，`quick_mode` 為 false，切分張數為 54,000／6,000／10,000。
6. 查看圖表及指標，下載整份 ZIP 或確認 Drive 中的檔案完整。
7. 在 `results/` 建立該次 Colab 實驗資料夾，複製成果並寫一份摘要；與本機 RTX 實測分開保存。

建議資料夾名稱以日期及簡短實驗名稱區分，例如 `results/colab_gpu_YYYYMMDD_baseline/`。這是命名範例，`YYYYMMDD` 請改成實際日期。

## 實驗證據對照

每一輪保留獨立成果資料夾，使用下列檔案追溯設定與結果：

| 項目 | 證據來源 |
| --- | --- |
| 實驗名稱、目的與輸出路徑 | 該輪 `summary.md` |
| 執行日期、裝置與 GPU 名稱 | `environment.json` 及硬體檢查紀錄 |
| Python、PyTorch、torchvision、CUDA | `environment.json` |
| quick_mode、訓練／驗證／測試張數 | `metrics.json` |
| seed、epochs、batch size、learning rate | `config.json` |
| 最佳 epoch、最佳驗證 loss、最終測試 loss 與 accuracy | `metrics.json` |
| 每個 epoch 的學習變化 | `history.csv`、`training_curves.png` |
| train_seconds 及計時範圍 | `metrics.json`、`summary.md` |
| 類別混淆與錯誤影像 | `confusion_matrix.png`、`misclassified.png` |
| 限制、失敗或中斷情況 | 該輪 `summary.md` |

`test_accuracy` 在 `metrics.json` 中是 0～1 的比例，轉成百分比時乘以 100。保留適當小數位，並確保表格與原始檔一致。

## 如何做公平的設定比較

初次先完成預設基準實驗。之後每次只更改一項，例如：

| 實驗 | epochs | batch size | learning rate | 目的 |
| --- | ---: | ---: | ---: | --- |
| 基準 | 5 | 128 | 0.001 | 建立預設流程的實際紀錄 |
| 學習率比較 | 5 | 128 | 0.0005 | 觀察更新步伐的影響 |
| 訓練長度比較 | 10 | 128 | 0.001 | 觀察學習與過擬合趨勢 |

這是**實驗設計範例**，沒有預設哪一組會較好。保持資料切分、模型、前處理與其他設定一致，依驗證 loss 比較，記錄所有嘗試；不要只保留最好的數字。

主流程每次執行都會對該次最佳 checkpoint 進行一次測試。若執行多輪比較，應先訂好比較項目，以驗證結果選擇設定，不要查看測試分數後再決定下一輪如何調參；反覆使用測試分數選模型會破壞獨立評估的用途。[官方模型評估說明](https://scikit-learn.org/stable/modules/cross_validation.html)

## CPU／GPU 時間比較

此項為選用延伸，不是完成教學的必要條件。若進行比較，請：

- 使用相同模型、資料量、epochs、batch size 與前處理，記錄各自硬體。
- 排除資料下載、安裝套件及手動操作時間，或另外列出這些時間。
- 清楚說明是否包含每個 epoch 的驗證、成果畫圖及檔案保存。
- 對 CUDA 計時，在開始與結束邊界等待 GPU 工作完成，或使用 CUDA events。
- 避免以 CPU quick 的時間除以 GPU 完整訓練時間計算加速倍數。

GPU 執行具有非同步特性，未同步的時間測量可能不準確。[PyTorch CUDA 官方說明](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

本專案先前 CPU 紀錄使用 `+cpu` 版 PyTorch，RTX 4060 實驗改用支援 CUDA 的版本；兩次軟體環境不同，且各只有單次執行，因此本報告不將兩者時間直接解讀為純硬體加速倍數。

未來若設計受控比較，應先統一軟體版本及工作量，再定義「CPU 時間 ÷ GPU 時間」的加速倍數，附上兩次完整設定與計時範圍。單次觀察不能推論到所有 GPU 或所有神經網路。

## 圖表討論提示

**學習曲線**：訓練與驗證 loss 是否一起下降？驗證 loss 最低位於何時？是否出現訓練改善而驗證惡化？不要只因為最後一個 epoch 較晚，就宣稱它是最佳模型。

**混淆矩陣**：縱軸為真實類別、橫軸為預測類別。先看對角線，再找非對角線較大的格子，搭配每類樣本數與 `per_class` 指標解讀。

**錯誤影像**：列出真實類別、預測類別與可見特徵。輪廓相似等解釋應標示為推測，除非另有實驗支持。沒有錯誤圖片時，程式不產生 `misclassified.png`，不代表輸出缺漏。

## 重現性與保存

固定 seed 有助降低初始化與資料順序造成的變異，仍不保證跨版本、硬體或 CPU/GPU 得到完全相同結果。[PyTorch 重現性說明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

Colab 的暫存虛擬機器會被回收；保存筆記本並不等於保存模型及其他輸出。每次實驗完成後，確認 Drive 或下載 ZIP 中同時包含設定、環境、歷史紀錄、指標、模型與圖表。[Colab 官方 FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)

若訓練中斷，記錄中斷位置與已保存的檔案。不要將不完整的歷史紀錄與另一輪測試分數合併成同一次實驗。
