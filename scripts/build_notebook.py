"""Build the self-contained Colab tutorial from the shared training source.

Only Python's standard library is needed to generate the notebook. Run this
script whenever src/training.py changes; the committed notebook is the result.
"""

import argparse
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
    required = {"imports", "data", "model", "engine", "outputs", "runner"}
    if missing := required - sections.keys():
        raise ValueError(f"Missing training sections: {sorted(missing)}")
    return sections


def build_notebook(source_path=DEFAULT_SOURCE):
    """Return a deterministic nbformat 4 notebook with no executed outputs."""
    sections = read_sections(source_path)
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

        這份期中報告以 **Python、PyTorch 與 Fashion-MNIST** 示範完整的神經網路實驗。
        請由上往下依序執行，每個程式格都已包含在本筆記本，不必先複製 GitHub 專案。

        完成後，你將能啟用並確認 GPU、切分資料、建立 CNN、訓練與挑選最佳模型，
        最後評估測試集、閱讀圖表、保存權重並載入模型進行推論。
        本筆記本沒有預填執行成果；準確率與耗時由你實際執行後產生。
    """)
    add("markdown", "runtime-instructions", """
        ## 1. 開啟 Colab 並啟用 GPU

        1. 在 [Google Colab](https://colab.research.google.com/) 選擇「檔案 → 上傳筆記本」，
           上傳 `colab_gpu_training.ipynb`，或直接從自己的 GitHub repository 開啟此檔。
        2. 選擇「執行階段 → 變更執行階段類型」，將硬體加速器設為 **GPU**，再連線。
           介面名稱可能隨語言或版本略有不同。
        3. 從第一個程式格開始依序執行；變更硬體後要重新執行全部程式格。

        Colab 的 GPU 型號、可用額度與連線時間會變動，免費資源不保證供應。
        本教學預設 `REQUIRE_GPU = True`：未偵測到 CUDA 時會停止並給出提示。
        若要先在 CPU 檢查流程，可在下方參數格明確改成 `False`，並在報告標示 **CPU**；
        CPU 執行不能當作 Colab GPU 實驗結果。
        請參考 [Colab 官方資源限制說明](https://research.google.com/colaboratory/faq.html#resource-limits)。

        Colab 預裝的 PyTorch 與 torchvision 應保留。下格只補裝缺少的 NumPy 或 Matplotlib，
        不重裝 CUDA、torch 或 torchvision；若這兩個主要套件缺少，請先確認你使用的是標準 Colab Python 執行階段。
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
    add("markdown", "data-design", """
        ## 2. 資料與可重現的實驗設計

        Fashion-MNIST 是 28 × 28 灰階服飾影像，共十類。原始 60,000 張訓練影像固定以
        seed 42 切為 **54,000 張訓練、6,000 張驗證**；官方 **10,000 張測試影像**留到模型挑選完才評估。
        訓練集更新權重，驗證集用來挑選 checkpoint，測試集用來估計最後模型的表現。

        每張影像先轉為 `[1, 28, 28]` 浮點張量，再以 `(像素值 − 0.5) / 0.5` 正規化，
        將 0～1 轉為約 −1～1。這是預先固定的常數，不從測試集估計統計量。
        固定種子有助於重現切分與訓練；不同硬體、套件版本仍可能產生數值差異。

        下格定義共用資料函式，尚未開始訓練。類別保留英文名稱，避免圖表中文字型缺失。
    """)
    add("code", "shared-data", sections["data"])
    add("markdown", "parameter-guide", """
        ## 3. 設定參數與保存位置

        預設為 **5 epochs、batch size 128、Adam 學習率 0.001**。
        `QUICK = False` 使用完整資料；改成 `True` 時，訓練／驗證／測試最多只用 2,048／512／512 張，
        且 `Config` 會把 epochs 設為 1 來檢查流程，結果不能代表完整實驗。
        想比較參數時，請保持資料切分與測試流程一致，以驗證集決定設定，最後才看測試成績。

        Colab 暫存磁碟可能隨執行階段結束而清除。可把 `USE_GOOGLE_DRIVE` 改成 `True`，
        依畫面授權掛載自己的 Drive，讓結果直接寫入 `MyDrive/colab_gpu_training`；資料下載仍放在本機暫存。
        每次執行本格會建立新的時間戳記資料夾，避免沿用先前的 checkpoint。
        若要重跑完整實驗，請從本格開始依序重新執行。
    """)
    add("code", "parameters", """
        REQUIRE_GPU = True
        QUICK = False
        USE_GOOGLE_DRIVE = False
        EPOCHS = 5

        CONFIG = Config(
            epochs=EPOCHS,
            batch_size=128,
            learning_rate=0.001,
            seed=42,
            val_size=6000,
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
        print("參數：", asdict(CONFIG))
        print("成果資料夾：", RUN_DIR.resolve())
    """, tags=["parameters"])
    add("markdown", "device-guide", """
        ## 4. 確認真正使用的裝置

        `torch.cuda.is_available()` 是判斷 PyTorch 能否使用 CUDA 的依據；只選取 Colab GPU 選項並不足夠。
        下格會印出實際裝置、GPU 名稱與套件版本。報告請保留這些環境資訊。
        稍後模型會用 `model.to(device)` 移到此裝置，每個 batch 的影像與標籤也必須移到同一裝置。

        如果這格因缺少 GPU 停止：確認加速器設定並重新連線，或等可用資源恢復後再試；
        要明確採用 CPU 備用模式，回到參數格設定 `REQUIRE_GPU = False` 再依序重跑。
    """)
    add("code", "select-device", """
        seed_everything(CONFIG.seed)
        device = get_device(require_gpu=CONFIG.require_gpu)
        print(json.dumps(environment_info(device), ensure_ascii=False, indent=2))
        print("本次模式：", "快速流程檢查" if CONFIG.quick else "完整資料實驗")
    """)
    add("markdown", "loader-guide", """
        ## 5. 下載、切分並預覽訓練影像

        第一次執行會下載 Fashion-MNIST；後續會重用 `DATA_DIR` 的檔案。
        DataLoader 將資料分成 batch，訓練資料會打亂，驗證與測試資料不打亂。
        下方只預覽訓練集，並用 `image * 0.5 + 0.5` 還原顯示亮度。
        請確認影像、類別與切分數量合理；快速模式會顯示縮小後的實際數量。
    """)
    add("code", "load-and-preview", """
        train_loader, val_loader, test_loader = make_loaders(CONFIG, data_dir=DATA_DIR)
        split_sizes = {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
        print("資料切分：", split_sizes)

        fig, axes = plt.subplots(3, 4, figsize=(9, 7))
        for index, axis in enumerate(axes.flat):
            image, label = train_loader.dataset[index]
            axis.imshow((image.squeeze(0) * 0.5 + 0.5).numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(CLASS_NAMES[int(label)])
            axis.axis("off")
        fig.suptitle("Training samples")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    """)
    add("markdown", "architecture", """
        ## 6. 建立卷積神經網路（CNN）

        卷積層學習局部影像特徵，池化縮小空間大小，ReLU 提供非線性，Dropout 在訓練時減少過度依賴少數特徵。
        最後輸出十個 **logits**。`CrossEntropyLoss` 直接接受 logits，所以模型最後不加 Softmax。

        | 階段 | 單張影像的形狀 |
        | --- | --- |
        | 輸入 | 1 × 28 × 28 |
        | Conv 1→16（3×3、padding 1）＋ReLU＋2×2 MaxPool | 16 × 14 × 14 |
        | Conv 16→32（3×3、padding 1）＋ReLU＋2×2 MaxPool | 32 × 7 × 7 |
        | 展平 | 1,568 |
        | Linear＋ReLU＋Dropout | 128 |
        | Linear | 10 |

        模型共有 **206,922 個可訓練參數**；下格會計算確認。`[batch, 1, 28, 28]` 輸入應得到 `[batch, 10]`。
    """)
    add("code", "shared-model", sections["model"])
    add("code", "create-model", """
        model = FashionCNN().to(device)
        print(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"可訓練參數：{parameter_count:,}")
        model.eval()
        with torch.no_grad():
            example_input = torch.zeros(2, 1, 28, 28, device=device)
            example_output = model(example_input)
        print("輸入形狀：", tuple(example_input.shape), "輸出形狀：", tuple(example_output.shape))
    """)
    add("markdown", "training-guide", """
        ## 7. 訓練與驗證

        一個 epoch 代表走過一次訓練資料。每個 batch 依序執行：
        **將張量移至裝置 → 清除梯度 → 前向傳播 → 計算 loss → 反向傳播 → Adam 更新權重**。
        `model.train()` 啟用訓練行為；驗證時用 `model.eval()` 與 `torch.inference_mode()`，不更新參數。

        每輪分別記錄訓練與驗證的 loss、accuracy。平均 loss 依樣本數加權，避免最後一個較小 batch 被算得太重。
        `fit` 只依 **最低驗證 loss** 保存 `best_model.pt`，不查看測試成績來挑選模型。
        訓練函式的原始碼放在下格，可直接看到 `images.to(device)` 與 `labels.to(device)`。
    """)
    add("code", "shared-engine", sections["engine"])
    add("code", "fit", """
        history, best_epoch, train_seconds = fit(
            model, train_loader, val_loader, CONFIG, device, RUN_DIR
        )
        print(f"最低驗證 loss 位於第 {best_epoch} 輪；訓練耗時 {train_seconds:.2f} 秒。")
    """)
    add("markdown", "checkpoint-guide", """
        ## 8. 還原最佳模型

        最後一輪不一定是驗證表現最佳的一輪。評估測試集前，先載入同一實驗資料夾中保存的最佳權重。
        `map_location=device` 讓權重載入目前裝置；`weights_only=True` 用於本專案儲存的基本資料與張量 checkpoint。
        此處只載入自己剛產生的檔案。
    """)
    add("code", "restore-best", """
        checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("已還原第", checkpoint["epoch"], "輪，驗證 loss =", checkpoint["val_loss"])
    """)
    add("markdown", "evaluation-guide", """
        ## 9. 最後一次測試與圖表分析

        下方 `save_artifacts` **對測試集評估一次**，再以同一份預測製作混淆矩陣與預測圖。
        Accuracy 是分類正確的比例；loss 也反映預測信心，兩者需一起閱讀。
        混淆矩陣的 **列是真實類別、欄是預測類別**，對角線以外的數量能指出容易混淆的服飾。

        訓練 loss 持續下降但驗證 loss 上升，可能出現過擬合。請據驗證曲線討論原因，
        不要看到測試成績後反覆修改超參數、再挑最高測試分數。
        以下所有數字與圖表都由本次實際執行產生；快速模式的測試樣本也較少。

        成果包含 `history.csv`、`metrics.json`、`environment.json`、`config.json`、`best_model.pt`，
        以及訓練曲線、混淆矩陣、預測圖；有錯誤預測時另存 `misclassified.png`。
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
        ## 10. 載入權重進行單張推論

        新建相同結構的模型並載入權重，就能恢復已訓練的模型。
        下格從已完成最終評估的測試資料中取一張影像，示範 `unsqueeze(0)` 加上 batch 維度，
        然後以 `argmax` 取最高分的類別。這只是單張推論示範，不重新計算測試指標。
        預測錯誤也應如實保留並討論；單張成功案例不能代表整體準確率。
    """)
    add("code", "reload-and-predict", """
        reloaded_model = FashionCNN().to(device)
        saved_checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=device, weights_only=True)
        reloaded_model.load_state_dict(saved_checkpoint["model_state_dict"])
        reloaded_model.eval()
        sample_image, sample_label = test_loader.dataset[0]
        with torch.no_grad():
            sample_logits = reloaded_model(sample_image.unsqueeze(0).to(device))
            sample_prediction = int(sample_logits.argmax(dim=1).item())
        print("真實類別：", CLASS_NAMES[int(sample_label)])
        print("預測類別：", CLASS_NAMES[sample_prediction])
    """)
    add("markdown", "export-guide", """
        ## 11. 保存與下載實驗成果

        先確認上方列出的檔案已產生。若使用暫存磁碟，離開前可把 `DOWNLOAD_RESULTS` 改成 `True`，
        將本次資料夾壓縮後下載到自己的電腦。Drive 使用者也可直接在 Drive 中取回成果。
        提交報告時請記錄 seed、實際切分大小、epochs、最佳輪次、測試指標、耗時與 GPU 型號，
        並附圖說明觀察到的現象；本機 CPU 驗證與 Colab GPU 實驗需分開標示。

        你也可以在 Colab 選擇「檔案 → 下載 → 下載 .ipynb」保存執行輸出，再上傳到自己的 GitHub。
        分享前檢查筆記本內是否含有私人 Drive 路徑或其他個人資訊。
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
        ## 參考資料與延伸練習

        - [Google Colab 官方 FAQ：GPU 與資源限制](https://research.google.com/colaboratory/faq.html)
        - [PyTorch 官方 Quickstart：資料、模型、訓練與保存](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
        - [torchvision 官方 FashionMNIST 文件](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
        - [PyTorch 官方 torch.load 文件](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
        - [Fashion-MNIST 資料集與類別說明](https://github.com/zalandoresearch/fashion-mnist)

        延伸時每次只改一項設定，例如學習率、Dropout 或 epoch 數，用驗證集比較並記錄理由。
        完整報告應區分「實際觀察」、「可能原因」與「未來待驗證的假設」。
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
