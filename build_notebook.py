"""Build a self-contained, unexecuted Colab notebook from train.py (stdlib only)."""

import hashlib
import json
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    source = (root / 'train.py').read_text(encoding='utf-8')
    cells = []

    def add(kind, text):
        cell = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True),
                'id': hashlib.sha256(text.encode()).hexdigest()[:12]}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# 在 Google Colab 重現 MNIST 訓練與結果對照

參考[中文文章](https://zhuanlan.zhihu.com/p/379291795)與[原作者 Notebook 固定版本](https://github.com/radathan1/Running-Deep-Learning-Model-on-GPU/blob/f7b4405ad1bfcfcc994f00a964904d1d5e1544e7/Running_Deep_Learning_Model_on_GPU%20%281%29.ipynb)，獨立實作相同的 MNIST、784→32→10 網路與兩階段 SGD。

請選「執行階段 → 變更執行階段類型 → GPU」，再「全部執行」。CPU 備援仍在 Colab 雲端，並非本機訓練。本 Notebook 自含完整程式，不需先上傳 GitHub。

這份範本沒有執行輸出；實際執行版本另存為 `mnist_colab_executed.ipynb`。原作者數值會明標為作者已儲存的驗證結果，與本次實跑分開。
''')
    add('markdown', '''## 1. 檢查雲端環境
使用 Colab 預載套件，保留其 CUDA 相容組合；不自動覆蓋 PyTorch。執行輸出與 `environment.json` 會記錄本次實際套件版本、GPU 型號及裝置。
''')
    add('code', '''import platform
import torch
import torchvision
print('Python:', platform.python_version())
print('PyTorch:', torch.__version__, 'Torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Colab CPU')
''')
    add('markdown', '''## 2. 完整實作
下格將完整程式寫成 `train.py`，與 GitHub 同名檔案相同。模型先計算 10 個 logits，以交叉熵求 loss，再由反向傳播與 SGD 更新權重。`evaluate()` 同時計算依樣本數加權與原作者批次平均指標，並關閉梯度。
''')
    add('code', '%%writefile train.py\n' + source)
    add('markdown', '''## 3. 訓練、驗證與獨立測試
60,000 張官方訓練資料拆成 50,000 訓練、10,000 驗證，另保留官方測試 10,000 張。固定種子 42；同一模型先以 lr=0.5 訓練 5 輪，再以 lr=0.1 訓練 5 輪。

原文的 `val_acc` 是批次指標平均；最後只有 16 筆的批次也佔 1/40，與按 10,000 筆樣本加權不同。本程式兩者都列出。測試資料僅在全部訓練後評估。
''')
    add('code', '''import importlib
import train
importlib.reload(train)
DEVICE = 'auto'  # 有 CUDA 就使用 GPU；否則仍在 Colab CPU 執行
OUTPUT_DIR = 'results'
metrics = train.run_experiment(device=DEVICE, output_dir=OUTPUT_DIR, seed=42, threads=2)
''')
    add('markdown', '''## 4. 閱讀結果
學習曲線包含原作者基準與本次驗證結果；混淆矩陣使用獨立測試集。預測範例固定取前 20 筆，錯誤案例固定取前 20 筆誤判，不挑選最漂亮的結果。
''')
    add('code', '''from pathlib import Path
import json
from IPython.display import display, Image
print(json.dumps(metrics, ensure_ascii=False, indent=2))
for filename in ['training_curves.png', 'confusion_matrix.png', 'predictions.png', 'mistakes.png']:
    display(Image(filename=str(Path(OUTPUT_DIR) / filename)))
''')
    add('markdown', '''## 5. 匯出實驗檔案
結果壓縮檔含 CSV、JSON、圖表與模型權重，另含 `train.py`。執行後點選「下載實驗結果 ZIP」。Notebook 本身請使用「檔案 → 下載 → 下載 .ipynb」，保留實際輸出。Colab 工作階段結束前需下載結果，僅分享 Notebook 不會自動分享執行階段檔案。
''')
    add('code', '''from zipfile import ZipFile, ZIP_DEFLATED
from base64 import b64encode
from IPython.display import HTML
archive_path = Path('mnist_colab_results.zip')
with ZipFile(archive_path, 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(Path(OUTPUT_DIR).iterdir()):
        if path.is_file() and path.suffix != '.zip':
            archive.write(path, arcname='results/' + path.name)
    archive.write('train.py')
print('Export ready:', archive_path.name)
encoded_archive = b64encode(archive_path.read_bytes()).decode('ascii')
display(HTML('<a download="mnist_colab_results.zip" href="data:application/zip;base64,' + encoded_archive + '">下載實驗結果 ZIP</a>'))
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
                'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                             'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                             'colab': {'provenance': [], 'name': 'mnist_colab.ipynb'},
                             'training_source_sha256': hashlib.sha256(source.encode()).hexdigest()}}
    path = root / 'notebooks' / 'mnist_colab.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Created {path.name}: {len(cells)} cells')


if __name__ == '__main__':
    build()
