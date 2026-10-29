# 指定文章與本專案的來源對照

本專案「**完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型**」以[安大叔的知乎文章](https://zhuanlan.zhihu.com/p/379291795)為主要依據。原文發布於 2021-06-09 18:17，教學任務是 MNIST 手寫數字分類，實作重點是 GPU／CPU 選擇，以及將資料與模型放到相同裝置。

另核對[英文原文](https://www.analyticsvidhya.com/blog/2021/05/a-complete-hands-on-guide-to-train-your-neural-network-model-on-google-colab-gpu/)（Adwait Dathan，2021-05-28）與其文末提供的[作者 GitHub notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)。本文件說明保留、修正與新增內容，不複製文章全文。

## 11 步主線對照

下表的步驟順序沿用知乎文章；[本專案筆記本](../notebooks/colab_gpu_training.ipynb)亦按此主線編排。程式函式位於 [src/training.py](../src/training.py)。

| 原文步驟 | 主要內容 | 本專案位置與處理 |
| --- | --- | --- |
| 1 | 設定 Colab GPU 執行階段 | 筆記本步驟 1；保留操作，資源說明依現行 FAQ 更新。 |
| 2 | 匯入 PyTorch 等套件 | 筆記本步驟 2；保留框架，另列實際版本。 |
| 3 | MNIST、ToTensor、訓練／驗證切分 | `make_loaders`；保留 50,000／10,000，新增 seed 42。 |
| 4 | 批次資料載入 | `make_loaders`；保留訓練 128、驗證 256；驗證不打亂、Windows workers 0。 |
| 5 | 建立 Mnistmodel | `MnistModel`；保留 784→32→10、ReLU、交叉熵。 |
| 6 | GPU 可用時選 CUDA，否則 CPU | `get_default_device`；另以 `get_device` 支援必須使用 GPU 的選項。 |
| 7 | 搬移張量與模型 | `to_device`；保留遞迴處理資料與 `non_blocking=True` 概念。 |
| 8 | 包裝 DataLoader 自動搬移每批資料 | `DeviceDataLoader`；修正文章的名稱與縮排展示問題。 |
| 9 | 定義訓練與驗證 | `fit`、`evaluate`；保留 SGD，修正模式、無梯度與樣本加權。 |
| 10 | 實例化模型並放至選定裝置 | `run_experiment` 與筆記本步驟 10。 |
| 11 | 訓練與驗證 | `fit` 與筆記本步驟 11；依作者 notebook 保留兩階段共 10 輪。 |

原文引言提到 VGG16，作為大型神經網路運算需求的例子；真正示範的是單隱藏層全連接模型，沒有卷積層。

原文將訓練與驗證方法放在模型類別內，本專案將這些流程整理為獨立函式，供本機入口與筆記本共用；`MnistModel` 保留前向傳播的相同架構。accuracy 由評估函式累計正確數與總樣本數，不另外宣稱存在同名輔助函式。

## 作者 notebook 的超參數依據

作者 notebook 的程式儲存格 33 以 SGD 訓練 5 輪、learning rate 0.5；儲存格 34 再以相同模型續訓 5 輪、learning rate 0.1。兩次 `fit` 各建立新的 SGD，沒有 momentum。本專案將此流程整理成預設 10 輪，於第 6 輪切換階段並重建 optimizer，且將每輪 learning rate 寫入歷史紀錄。[作者原始 notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)

作者在儲存格 32 先評估未訓練模型，本專案保留此觀察並保存為 `initial_validation.json`。作者程式沒有以官方獨立測試集提供最終測試，也沒有保存最佳驗證 checkpoint；這兩項不被描述成原文原有功能。

## 必要的程式修正

| 問題 | 修正及理由 |
| --- | --- |
| 文章的 `random_split` 展示缺少括號 | 依可執行語法修復；作者 notebook 可供核對。 |
| DeviceDataLoader 大小寫、名稱與縮排不一致 | 統一類別名稱及方法縮排，不改變逐批搬移概念。 |
| 文章使用 accuracy 卻未完整展示定義 | 依作者 notebook 與正確分類比例補齊計算。 |
| 原展示未明確切換 train／eval，驗證仍建立梯度圖 | 訓練使用 `model.train()`；驗證及推論使用 `model.eval()` 與無梯度範圍，符合 PyTorch 評估流程。 |
| 原展示將不同大小的批次指標直接平均 | 改按樣本數彙總 loss 及正確分類數，避免最後的小批次與完整批次等權。 |
| 原驗證 DataLoader 打亂順序 | 改為不打亂，便於穩定檢查與分析；不影響驗證集用途。 |
| 原 workers 為 2 | Windows 預設 0，Colab 可選 2；避免平台差異妨礙入門操作。 |

train／eval 與驗證時不計算梯度的用途，可參考 [PyTorch 官方訓練教學](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

## 明確新增的實驗補強

- **可重做的設定**：seed 42、設定與環境檔，並記錄每輪學習率。
- **獨立測試**：官方訓練資料切成 50,000／10,000；另保留 `train=False` 的 10,000 張測試圖，待模型選定後評估。[MNIST API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- **最佳模型**：以驗證 loss 選出 checkpoint，保存後可載入推論。[PyTorch 模型保存與載入](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- **結果分析**：訓練曲線、測試混淆矩陣及 CSV、每類指標、預測與錯例。
- **執行證據**：本機 RTX 4060 的實際 CUDA 裝置、硬體、時間、記憶體及模型載入檢查；本機與 Colab 雲端分開標示。
- **成果保存**：筆記本提供 Drive 與 ZIP 操作，減少暫存環境結束後失去檔案的情況。

這些補強讓成果可以檢查與還原，但不將新增內容歸因於原作者，也不使用原作者的驗證分數充當本專案結果。

## 時效性更新

原文的 Tesla K80 及使用時間描述反映 2021 年的教學背景。現行 Colab 的 GPU 型號、供應與使用限制會變動，並非固定或保證；因此 README 與筆記本要求檢查當次實際裝置，並依[官方 FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)說明保存及資源限制。

## 主要來源

1. [指定知乎文章](https://zhuanlan.zhihu.com/p/379291795)，安大叔，2021-06-09。
2. [英文原文](https://www.analyticsvidhya.com/blog/2021/05/a-complete-hands-on-guide-to-train-your-neural-network-model-on-google-colab-gpu/)，Adwait Dathan，2021-05-28。
3. [原作者 notebook](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)。
