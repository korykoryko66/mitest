"""Validate the generated notebook; optionally execute an isolated CPU quick run.

Static validation only needs the standard library. Execution needs the runtime
packages from requirements.txt and permission to download MNIST if the
specified data directory does not already contain it.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

from build_notebook import DEFAULT_NOTEBOOK, DEFAULT_SOURCE, TITLE, build_notebook


def validate_notebook(notebook, source_path=DEFAULT_SOURCE):
    """Check nbformat structure, compile every code cell, and detect source drift."""
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        raise ValueError("Expected notebook format 4.5")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not 20 <= len(cells) <= 45:
        raise ValueError("The tutorial should contain 20 to 45 cells")
    if not cells or TITLE not in "".join(cells[0].get("source", [])):
        raise ValueError("The required report title is missing")
    if notebook.get("metadata", {}).get("accelerator") != "GPU":
        raise ValueError("The notebook must request a Colab GPU accelerator")
    expected_steps = [
        "step-01-colab-gpu", "step-02-imports", "step-03-mnist-split", "step-04-data-loader",
        "step-05-mnist-model", "step-06-default-device", "step-07-to-device",
        "step-08-device-loader", "step-09-fit-evaluate", "step-10-create-model",
        "step-11-train-validate",
    ]
    actual_steps = [cell.get("id", "") for cell in cells if cell.get("id", "").startswith("step-")]
    if actual_steps != expected_steps:
        raise ValueError("The teacher-assigned article's eleven steps must appear in order")
    ids = set()
    parameter_cells = 0
    device_definition_cells = 0
    code_cells = 0
    for index, cell in enumerate(cells):
        cell_id = cell.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) or cell_id in ids:
            raise ValueError(f"Invalid or duplicate cell id in cell {index}: {cell_id}")
        ids.add(cell_id)
        if not isinstance(cell.get("metadata"), dict):
            raise ValueError(f"Cell {index} has invalid metadata")
        source = cell.get("source")
        if not isinstance(source, list) or not all(isinstance(line, str) for line in source):
            raise ValueError(f"Cell {index} source must be a list of strings")
        if cell["cell_type"] == "code":
            code_cells += 1
            if cell.get("execution_count") is not None or cell.get("outputs") != []:
                raise ValueError(f"Cell {index} must not include stored execution outputs")
            compile("".join(source), f"notebook-cell-{index}-{cell_id}", "exec")
        elif cell["cell_type"] != "markdown":
            raise ValueError(f"Unsupported cell type in cell {index}")
        if "parameters" in cell["metadata"].get("tags", []):
            parameter_cells += 1
        if "device-selection-definitions" in cell["metadata"].get("tags", []):
            device_definition_cells += 1
    if parameter_cells != 1:
        raise ValueError("Expected exactly one parameters cell")
    if device_definition_cells != 1:
        raise ValueError("Expected exactly one device-selection definitions cell")
    if notebook != build_notebook(source_path):
        raise ValueError("Notebook differs from the shared source. Run scripts/build_notebook.py again.")
    return code_cells


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def execute_notebook(notebook, data_dir, work_dir):
    """Execute the embedded code only, with explicit CPU/quick validation overrides."""
    data_dir = Path(data_dir).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if (work_dir / "src").exists():
        raise ValueError("Use an isolated --work-dir without the repository's src directory")
    namespace = {"__name__": "__main__"}
    previous_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        with working_directory(work_dir):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                print(f"Executing cell {index + 1}/{len(notebook['cells'])}: {cell['id']}", flush=True)
                exec(compile("".join(cell["source"]), f"notebook-cell-{index}-{cell['id']}", "exec"), namespace)
                if "parameters" in cell["metadata"].get("tags", []):
                    namespace["REQUIRE_GPU"] = False
                    namespace["QUICK"] = True
                    namespace["USE_GOOGLE_DRIVE"] = False
                    namespace["EPOCHS"] = 1
                    namespace["CONFIG"] = namespace["Config"](epochs=1, quick=True, require_gpu=False)
                    namespace["DATA_DIR"] = data_dir
                    namespace["OUTPUT_ROOT"] = work_dir / "outputs"
                    namespace["RUN_DIR"] = namespace["OUTPUT_ROOT"] / datetime.now(timezone.utc).strftime(
                        "notebook_cpu_validation_%Y%m%dT%H%M%S_%fZ"
                    )
                    namespace["RUN_DIR"].mkdir(parents=True, exist_ok=False)
                    print("Validation override: CPU, one epoch, quick subset; this is not a Colab GPU result.", flush=True)
                if "device-selection-definitions" in cell["metadata"].get("tags", []):
                    # Apply after the shared definitions, so they cannot overwrite the CPU override.
                    namespace["get_default_device"] = lambda: namespace["torch"].device("cpu")
                    namespace["get_device"] = lambda require_gpu=False: namespace["torch"].device("cpu")
        run_dir = namespace["RUN_DIR"]
        required = {
            "history.csv", "metrics.json", "environment.json", "config.json", "best_model.pt",
            "initial_validation.json", "confusion_matrix.csv",
            "training_curves.png", "confusion_matrix.png", "predictions.png",
        }
        if missing := required - {path.name for path in run_dir.iterdir()}:
            raise ValueError(f"Execution finished but required artifacts are missing: {sorted(missing)}")
        print(f"Standalone CPU quick execution passed. Results: {run_dir}")
        return run_dir
    finally:
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--execute", action="store_true", help="Also run all embedded cells on CPU with one quick epoch")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_NOTEBOOK.parents[1] / "data")
    parser.add_argument("--work-dir", type=Path, help="Isolated execution directory; omitted means temporary, deleted after validation")
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    code_cells = validate_notebook(notebook, args.source)
    print(f"Notebook format, {code_cells} code cells, and shared-source consistency passed.")
    if args.execute:
        if args.work_dir is not None:
            execute_notebook(notebook, args.data_dir, args.work_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="colab_notebook_validation_") as folder:
                execute_notebook(notebook, args.data_dir, folder)
            print("Temporary execution directory was removed; use --work-dir to preserve validation artifacts.")


if __name__ == "__main__":
    main()
