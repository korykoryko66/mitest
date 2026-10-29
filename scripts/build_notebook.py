"""Build the article-aligned, self-contained Colab notebook from shared source.

Only Python's standard library is required. Regenerate the committed notebook
whenever src/training.py changes; no repository import is needed inside Colab.
"""

import argparse
import ast
import json
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "src" / "training.py"
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "colab_gpu_training.ipynb"
TITLE = "完整的動手指南：可在 Google Colab GPU 上訓練你的神經網路模型"


def read_sections(source_path=DEFAULT_SOURCE):
    """Read explicitly delimited sections without importing training packages."""
    source = Path(source_path).read_text(encoding="utf-8")
    markers = list(re.finditer(r"^# %% (\w+)\s*$", source, re.MULTILINE))
    sections = {}
    for index, marker in enumerate(markers):
        name = marker.group(1)
        if name in sections:
            raise ValueError(f"Duplicate training section: {name}")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        sections[name] = source[marker.end():end].strip() + "\n"
    required = {"imports", "data", "model", "device", "engine", "outputs", "runner"}
    if missing := required - sections.keys():
        raise ValueError(f"Missing training sections: {sorted(missing)}")
    return sections


def take_definitions(source, names):
    """Extract complete top-level definitions verbatim, retaining their order."""
    names = set(names)
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name in names]
    if missing := names - {node.name for node in nodes}:
        raise ValueError(f"Missing shared definitions: {sorted(missing)}")
    lines = source.splitlines(keepends=True)
    selected, removed = [], set()
    for node in nodes:
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)]) - 1
        end = node.end_lineno
        selected.append("".join(lines[start:end]).rstrip())
        removed.update(range(start, end))
    remaining = "".join(line for index, line in enumerate(lines) if index not in removed)
    return "\n\n\n".join(selected) + "\n", remaining.strip() + "\n"


def build_notebook(source_path=DEFAULT_SOURCE):
    """Return a deterministic nbformat 4 notebook with no stored execution outputs."""
    sections = read_sections(source_path)
    loader_definition, data_definitions = take_definitions(sections["data"], {"make_loaders"})
    device_selection, remaining_device = take_definitions(
        sections["device"], {"get_default_device", "get_device"}
    )
    to_device_definition, remaining_device = take_definitions(remaining_device, {"to_device"})
    device_loader_definition, remaining_device = take_definitions(remaining_device, {"DeviceDataLoader"})
    if ast.parse(remaining_device).body:
        raise ValueError("Unexpected definitions in the device section; update the notebook mapping")
    cells = []

    def add(kind, name, content, tags=None):
        cell = {
            "cell_type": kind,
            "id": name,
            "metadata": {"tags": tags} if tags else {},
            "source": (textwrap.dedent(content).strip() + "\n").splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "title", f"""
        # {TITLE}

        **主要依據：老師指定的 [知乎文章](https://zhuanlan.zhihu.com/p/379291795)。**
        本筆記本依文章的 **11 個步驟**安排，以 **MNIST 手寫數字、PyTorch 與兩層前饋神經網路**
        示範 Colab GPU 訓練；並對照英文原文與作者提供的 notebook 確認程式與超參數。

        核心實作為 **784 → 32 → 10、ReLU、SGD**：先用學習率 **0.5 訓練 5 輪**，
        再沿用同一組模型權重，以 **0.1 訓練 5 輪**，不使用 momentum。
        種子、完整測試集評估、最佳模型保存與圖表匯出是本專案另加的實驗管理功能。

        請由上往下執行。所有訓練程式已包含於本檔，不必複製 repository 或遠端匯入原始碼。
        本筆記本沒有預填執行成果；loss、準確率與耗時都由實際執行後產生。
    """)
    add("markdown", "step-01-colab-gpu", """
        ## 步驟 1：在 Google Colab 啟用 GPU

        1. 開啟 [Google Colab](https://colab.research.google.com/)，選擇「檔案 → 上傳筆記本」，
           上傳這份 `colab_gpu_training.ipynb`。
        2. 選擇「執行階段 → 變更執行階段類型」，將硬體加速器設成 **GPU**，再連線。
           介面名稱可能隨語言或版本略有不同。
        3. 從第一個程式格開始依序執行；變更硬體或重新連線後，需重新執行所有程式格。

        原文發布時的 GPU 型號與時數描述屬當時環境。現在 Colab 的 GPU 型號、用量限制與連線時間會變動，
        無法保證獲配某一型號或固定使用時數，請以 [Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html#resource-limits) 為準。
        本教學預設 `REQUIRE_GPU = True`，稍後會確認 CUDA 真正可用。
        若未取得 GPU 而要先以 CPU 練習，可在參數格改成 `False`，允許自動回退至 CPU；
        GPU 可用時仍會選擇 GPU。報告請註明實際裝置，CPU 驗證不能當作 GPU 實驗成果。
    """)
    add("markdown", "step-02-imports", """
        ## 步驟 2：匯入套件

        文章使用 PyTorch、torchvision 與 Matplotlib。下方保留 Colab 預裝的相容 **torch / torchvision**，
        只補裝缺少的 NumPy 或 Matplotlib；不無條件重裝 PyTorch 或 CUDA。
        隨後的共用匯入格也定義 `Config` 與環境紀錄工具，方便完整記錄這次實驗。
    """)
    add("code", "dependencies", """
        import importlib.util
        import subprocess
        import sys

        missing = [name for name in ("numpy", "matplotlib")
                   if importlib.util.find_spec(name) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        for name in ("torch", "torchvision"):
            if importlib.util.find_spec(name) is None:
                raise ModuleNotFoundError(
                    f"缺少 {name}。請使用標準 Colab Python 執行階段；"
                    "本機環境請依 README 安裝相容的 PyTorch / torchvision。"
                )
        print("套件檢查完成；保留目前的 PyTorch / torchvision。")
    """)
    add("code", "shared-imports", sections["imports"])
    add("markdown", "parameter-guide", """
        ### 執行參數與成果位置

        完整模式預設 **10 epochs、訓練 batch 128、驗證／測試 batch 256**。
        第 1～5 輪使用 SGD 學習率 0.5，第 6～10 輪改成 0.1，承接原模型權重。
        這對應作者 notebook 的兩次 `fit(5, ...)`；本專案以一次 `fit` 保存連續十輪的紀錄。

        `QUICK = True` 會使用少量資料並強制只跑一輪，用於檢查流程，不能代表完整的兩階段實驗。
        seed 42 是本專案新增的固定設定；不同硬體或套件版本仍可能出現數值差異。

        Colab 暫存檔可能隨執行階段結束而清除。可把 `USE_GOOGLE_DRIVE` 改為 `True`，
        依畫面授權掛載自己的 Drive，將成果寫入 `MyDrive/colab_gpu_training`。
        每次執行參數格會建立新的時間戳記資料夾，避免混用舊 checkpoint。
        重新實驗時，請從本格開始依序重跑。
    """)
    add("code", "parameters", """
        REQUIRE_GPU = True
        QUICK = False
        USE_GOOGLE_DRIVE = False
        EPOCHS = 10

        CONFIG = Config(
            epochs=EPOCHS,
            batch_size=128,
            val_batch_size=256,
            learning_rate=0.5,
            second_learning_rate=0.1,
            switch_epoch=5,
            hidden_size=32,
            seed=42,
            val_size=10000,
            num_workers=0,
            quick=QUICK,
            require_gpu=REQUIRE_GPU,
        )
        DATA_DIR = Path("data")
        OUTPUT_ROOT = Path("outputs")
        if USE_GOOGLE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")
            OUTPUT_ROOT = Path("/content/drive/MyDrive/colab_gpu_training")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_DIR = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("colab_%Y%m%dT%H%M%S_%fZ")
        RUN_DIR.mkdir(parents=True, exist_ok=False)
        seed_everything(CONFIG.seed)
        print("參數：", asdict(CONFIG))
        print("成果資料夾：", RUN_DIR.resolve())
    """, tags=["parameters"])
    add("markdown", "step-03-mnist-split", """
        ## 步驟 3：準備 MNIST 並切分訓練／驗證資料

        MNIST 每張影像是 **28 × 28 灰階手寫數字**，標籤為 0～9。
        依文章作法，僅使用 `transforms.ToTensor()` 將像素轉成 0～1 張量，**不另加 Normalize**。
        原始 60,000 張訓練影像分成 **50,000 張訓練、10,000 張驗證**。
        本專案以 seed 42 固定隨機切分，讓各次實驗使用相同資料。

        另外保留官方 **10,000 張測試影像**，不參與梯度更新、學習率調整或最佳模型挑選。
        測試集是本專案額外加入的最終評估；文章核心練習主要觀察驗證表現。
    """)
    add("code", "shared-data", data_definitions)
    add("markdown", "step-04-data-loader", """
        ## 步驟 4：建立 DataLoader

        DataLoader 將資料組成 mini-batch。訓練用 `batch_size=128` 並打亂順序，
        驗證／測試用 `batch_size=256` 且不打亂。
        下格第一次執行會下載 MNIST，之後重用 `DATA_DIR` 中的檔案。
        此時 DataLoader 仍在 CPU；步驟 8 會用 `DeviceDataLoader` 包裝，將每個 batch 自動移到裝置。

        預覽取自訓練集，影像仍是 `ToTensor()` 的 0～1 值，可直接顯示。
        請確認實際切分數量與影像標籤；快速模式會顯示縮小後的資料量。
    """)
    add("code", "shared-data-loaders", loader_definition)
    add("code", "load-and-preview", """
        train_loader, val_loader, test_loader = make_loaders(CONFIG, data_dir=DATA_DIR)
        split_sizes = {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
        print("資料切分：", split_sizes)
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for index, axis in enumerate(axes.flat):
            image, label = train_loader.dataset[index]
            axis.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"Label: {int(label)}")
            axis.axis("off")
        fig.suptitle("MNIST training samples")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    """)
    add("markdown", "step-05-mnist-model", """
        ## 步驟 5：定義 MnistModel

        依文章與作者 notebook 使用兩層前饋神經網路：

        | 階段 | 每張影像的維度 |
        | --- | --- |
        | 輸入灰階影像 | 1 × 28 × 28 |
        | 展平成向量 | 784 |
        | Linear 784→32，再經 ReLU | 32 |
        | Linear 32→10，輸出 logits | 10 |

        預設模型共有 **25,450 個參數**：`784×32 + 32 + 32×10 + 10`。
        `CrossEntropyLoss` 直接接受 logits，模型尾端不需要加 Softmax。
        下格先定義類別，步驟 10 才實際建立並移動模型。
    """)
    add("code", "shared-model", sections["model"])
    add("markdown", "step-06-default-device", """
        ## 步驟 6：以 get_default_device 選擇裝置

        文章的 `get_default_device()` 會在 CUDA 可用時選 GPU，否則回傳 CPU。
        本專案另加 `get_device(require_gpu=True)` 檢查：若本次實驗要求 GPU 卻沒有 CUDA，就停止並明確提示。
        這能避免把默默改用 CPU 的執行誤寫成 GPU 成果。

        若下格停止，請確認步驟 1 的 GPU 設定並重新連線，或稍後再試。
        若未取得 GPU 而要先以 CPU 練習，回到參數格將 `REQUIRE_GPU = False` 再依序重跑；
        這會允許 CPU 備用，GPU 可用時仍會選擇 GPU。
        只設定 Colab 選單不足以確認成功，應保留這格印出的實際裝置、GPU 名稱及套件版本。
    """)
    add("code", "shared-device-selection", device_selection, tags=["device-selection-definitions"])
    add("code", "select-device", """
        print("get_default_device()：", get_default_device())
        device = get_device(require_gpu=CONFIG.require_gpu)
        environment = environment_info(device)
        print(json.dumps(environment, ensure_ascii=False, indent=2))
        (RUN_DIR / "config.json").write_text(json.dumps(asdict(CONFIG), ensure_ascii=False, indent=2), encoding="utf-8")
        (RUN_DIR / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
        print("本次模式：", "快速流程檢查" if CONFIG.quick else "完整兩階段實驗")
    """)
    add("markdown", "step-07-to-device", """
        ## 步驟 7：定義 to_device

        資料和模型必須位於同一裝置。`to_device(data, device)` 會遞迴處理 list／tuple，
        讓包含影像與標籤的 batch 一起移到 GPU 或 CPU，也能移動模型。
        這把文章的裝置搬移概念整理為可重用工具，避免只移動模型、忘記移動資料。
    """)
    add("code", "shared-to-device", to_device_definition)
    add("code", "move-tensor-example", """
        example_tensor = torch.zeros(2, 1, 28, 28)
        example_tensor = to_device(example_tensor, device)
        print("範例張量所在裝置：", example_tensor.device)
    """)
    add("markdown", "step-08-device-loader", """
        ## 步驟 8：以 DeviceDataLoader 自動搬移 batch

        `DeviceDataLoader` 包裝原本的 DataLoader，在每次迭代時呼叫 `to_device`。
        如此訓練與驗證迴圈收到的影像、標籤就已位於選定裝置。
        下格明確包裝三個 loader，並檢查一個驗證 batch 的影像與標籤裝置。
    """)
    add("code", "shared-device-loader", device_loader_definition)
    add("code", "wrap-device-loaders", """
        train_loader = DeviceDataLoader(train_loader, device)
        val_loader = DeviceDataLoader(val_loader, device)
        test_loader = DeviceDataLoader(test_loader, device)
        images, labels = next(iter(val_loader))
        print("影像裝置：", images.device, "標籤裝置：", labels.device)
        print("影像形狀：", tuple(images.shape), "標籤形狀：", tuple(labels.shape))
    """)
    add("markdown", "step-09-fit-evaluate", """
        ## 步驟 9：定義 fit 與 evaluate

        `fit` 的 mini-batch 迴圈依序進行：**清除梯度 → 前向傳播 → 計算交叉熵 → 反向傳播 → SGD 更新**。
        每輪切換 `model.train()`／`model.eval()`；`evaluate` 以停用梯度的模式計算驗證 loss 與 accuracy，
        不更新權重。本專案依每個 batch 的樣本數加權平均 loss，讓最後一個較小 batch 的權重正確。

        作者 notebook 先以 `fit(5, 0.5, ...)` 訓練，再沿用同一個 model 呼叫 `fit(5, 0.1, ...)`。
        下方共用 `fit` 用連續十輪紀錄實作相同的 **5 輪 0.5 ＋ 5 輪 0.1**，
        不重新初始化權重，SGD 不使用 momentum，並將每輪學習率存入 `history.csv`。
        最低驗證 loss 的 checkpoint 保存是本專案新增功能，不使用測試集挑模型。
    """)
    add("code", "shared-engine", sections["engine"])
    add("markdown", "step-10-create-model", """
        ## 步驟 10：建立模型並移到選定裝置

        現在建立 `MnistModel(784, 32, 10)`，並以 `to_device` 移到與 batch 相同的裝置。
        建立前重新設定 seed，讓前面展示 batch 的操作不影響模型初始化。
        預設參數量應為 **25,450**；形狀檢查中，`[2, 1, 28, 28]` 輸入應得到 `[2, 10]` logits。
    """)
    add("code", "create-model", """
        seed_everything(CONFIG.seed)
        model = MnistModel(input_size=784, hidden_size=CONFIG.hidden_size, out_size=10)
        model = to_device(model, device)
        print(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"可訓練參數：{parameter_count:,}")
        with torch.no_grad():
            example_input = torch.zeros(2, 1, 28, 28, device=device)
            example_output = model(example_input)
        print("輸入形狀：", tuple(example_input.shape), "輸出形狀：", tuple(example_output.shape))
        print("模型所在裝置：", next(model.parameters()).device)
    """)
    add("markdown", "step-11-train-validate", """
        ## 步驟 11：先驗證未訓練模型，再訓練與驗證

        依文章先觀察未訓練模型的驗證表現，接著才更新權重。
        本專案把未訓練的 **loss／accuracy 純量**存為 `initial_validation.json`，方便和訓練後比較。
        完整模式會顯示連續十輪紀錄，前五輪學習率為 0.5，後五輪為 0.1。
        快速模式只檢查一輪，不涵蓋第二階段，不能用於聲稱完成文章的全部訓練。

        請看驗證 loss 與 accuracy 是否改善；若沒有改善，先檢查資料、裝置與學習率，
        不要用尚未評估的測試集調整參數。訓練結果由本次環境實測，不保證重現原文某個固定分數。
    """)
    add("code", "initial-validation", """
        initial = evaluate(model, val_loader, device)
        initial_validation = {"loss": float(initial["loss"]), "accuracy": float(initial["accuracy"])}
        (RUN_DIR / "initial_validation.json").write_text(
            json.dumps(initial_validation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("未訓練模型的驗證表現：", initial_validation)
        del initial
    """)
    add("code", "fit", """
        history, best_epoch, train_seconds = fit(
            model, train_loader, val_loader, CONFIG, device, RUN_DIR
        )
        print(f"最低驗證 loss 位於第 {best_epoch} 輪；訓練耗時 {train_seconds:.2f} 秒。")
    """)
    add("markdown", "final-evaluation-guide", """
        ## 本專案補充：還原最佳模型並評估官方測試集

        上面已完成文章的 11 個步驟。以下為本專案新增的實驗保存與分析功能。
        最後一輪不一定具有最低驗證 loss，先還原本次實驗資料夾的最佳 checkpoint，
        再對保留的官方測試集 **評估一次**；圖表與各類指標沿用同一份預測。

        `map_location=device` 讓權重載入目前裝置；`weights_only=True` 用於自己剛保存的基本資料與張量 checkpoint。
        此檔供模型推論使用，沒有保存可完整恢復訓練的 optimizer 狀態。

        成果包括 `initial_validation.json`、`history.csv`、`metrics.json`、`environment.json`、`config.json`、
        `best_model.pt`、`confusion_matrix.csv`，以及訓練曲線、混淆矩陣和預測圖；有錯誤時另存錯誤預測圖。
        混淆矩陣的 **列是真實數字、欄是預測數字**。請同時閱讀 loss、accuracy、混淆矩陣與誤分類案例。
    """)
    add("code", "restore-best", """
        checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = to_device(model, device)
        model.eval()
        print("已還原第", checkpoint["epoch"], "輪，驗證 loss =", checkpoint["val_loss"])
    """)
    add("code", "shared-outputs", sections["outputs"])
    add("code", "evaluate-and-display", """
        metrics = save_artifacts(
            model, test_loader, history, CONFIG, device, RUN_DIR,
            best_epoch, train_seconds, split_sizes,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        try:
            from IPython.display import Image, display
        except ImportError:
            Image = None
        for filename in ("training_curves.png", "confusion_matrix.png", "predictions.png", "misclassified.png"):
            image_path = RUN_DIR / filename
            if image_path.exists():
                if Image is not None:
                    display(Image(filename=str(image_path), width=900))
                else:
                    print("圖表：", image_path.resolve())
        print("成果檔案：", [path.name for path in sorted(RUN_DIR.iterdir())])
    """)
    add("markdown", "inference-guide", """
        ### 載入權重進行單張推論

        新建相同結構的模型並載入權重，即可恢復模型做推論。
        下格從已完成評估的測試資料取一張影像，以 `unsqueeze(0)` 加上 batch 維度，
        再用 `argmax` 取得預測數字。這是單張推論示範，不重新計算測試指標。
        預測錯誤也應如實保留；單張成功案例不代表整體準確率。
    """)
    add("code", "reload-and-predict", """
        reloaded_model = MnistModel(input_size=784, hidden_size=CONFIG.hidden_size, out_size=10)
        reloaded_model = to_device(reloaded_model, device)
        saved_checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=device, weights_only=True)
        reloaded_model.load_state_dict(saved_checkpoint["model_state_dict"])
        reloaded_model.eval()
        sample_image, sample_label = test_loader.dataset[0]
        with torch.no_grad():
            sample_logits = reloaded_model(to_device(sample_image.unsqueeze(0), device))
            sample_prediction = int(sample_logits.argmax(dim=1).item())
        print("真實數字：", int(sample_label))
        print("預測數字：", sample_prediction)
    """)
    add("markdown", "export-guide", """
        ### 保存與下載實驗成果

        若使用 Colab 暫存磁碟，離開前可把 `DOWNLOAD_RESULTS` 改成 `True`，壓縮並下載本次成果。
        Drive 使用者也可在自己的 Drive 取回檔案。
        報告應記錄實際資料量、seed、兩階段學習率、epochs、未訓練驗證指標、最佳輪次、測試指標、耗時及實際 GPU 型號。
        本機 CPU 驗證、快速模式和完整 Colab GPU 實驗需分開標示。

        在 Colab 選「檔案 → 下載 → 下載 .ipynb」可保存執行輸出，再將修訂後的 notebook 上傳自己的 GitHub。
        分享前請檢查輸出是否包含私人 Drive 路徑或其他個人資訊。
    """)
    add("code", "download-results", """
        DOWNLOAD_RESULTS = False
        if DOWNLOAD_RESULTS:
            import shutil
            from google.colab import files
            archive_path = shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR)
            files.download(archive_path)
        else:
            print("成果保存在：", RUN_DIR.resolve())
    """)
    add("markdown", "references", """
        ## 依據、對照與延伸

        - **老師指定的主要依據：**[知乎文章](https://zhuanlan.zhihu.com/p/379291795)。本筆記本的 11 步驟順序依此安排。
        - [英文原文：A Complete Hands-on Guide to Train Your Neural Network Model on Google Colab GPU](https://www.analyticsvidhya.com/blog/2021/05/a-complete-hands-on-guide-to-train-your-neural-network-model-on-google-colab-gpu/)
        - [作者 notebook：兩層 MnistModel 與 5 輪 0.5 ＋ 5 輪 0.1 SGD](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/main/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)
        - [Google Colab 官方 FAQ：當前 GPU 與資源限制](https://research.google.com/colaboratory/faq.html)
        - [torchvision 官方 MNIST 文件](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
        - [PyTorch 官方 torch.load 文件](https://docs.pytorch.org/docs/stable/generated/torch.load.html)

        文章中的排版與程式片段語法問題已整理為可依序執行的 Python；核心資料、模型、裝置搬移與兩階段 SGD 依作者 notebook。
        seed 42、樣本加權平均、停用梯度的評估、官方測試集、checkpoint 與成果匯出屬本專案補充。
        GitHub／本機專案另附 [文章對照說明](../docs/SOURCE_ALIGNMENT.md)，逐項區分來源核心與新增功能。
        延伸實驗請每次只改一項設定，以驗證集比較，並分清實際觀察與待驗證的假設。
    """)

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "name": "colab_gpu_training.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()
    notebook = build_notebook(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {args.output} ({len(notebook['cells'])} cells, no stored execution outputs)")


if __name__ == "__main__":
    main()
