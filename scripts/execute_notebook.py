"""
Robust in-process execution engine for Jupyter Notebook.
Executes code cells in sequential order, maintaining notebook namespace,
and records genuine runtime outputs directly into the .ipynb structure.
Usage:
    python scripts/execute_notebook.py
"""

import io
import sys
import os
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_output

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Joint_Input_Robustness_Experiment.ipynb"
SEPARATE_NOTEBOOK_PATH = WORKSPACE_ROOT / "Joint_Input_Robustness_Experiment.ipynb"


def execute_notebook():
    print(f"[*] Loading notebook from: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Initialize shared namespace with project root in path
    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
    }

    # Add project root to sys.path in current process
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    exec_count = 1
    total_code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"[*] Executing {total_code_cells} code cells sequentially...")

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        code = cell.source
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        cell_error = None
        try:
            exec(code, namespace)
        except Exception as e:
            cell_error = e
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_str = stdout_capture.getvalue()
        stderr_str = stderr_capture.getvalue()

        cell.outputs = []
        cell.execution_count = exec_count
        exec_count += 1

        if stdout_str:
            cell.outputs.append(new_output("stream", name="stdout", text=stdout_str))
        if stderr_str:
            cell.outputs.append(new_output("stream", name="stderr", text=stderr_str))

        if cell_error:
            err_name = type(cell_error).__name__
            err_val = str(cell_error)
            print(f"[!] Error in cell {exec_count-1}: {err_name}: {err_val}")
            cell.outputs.append(new_output(
                "error",
                ename=err_name,
                evalue=err_val,
                traceback=[f"{err_name}: {err_val}"]
            ))
            # Continue or break depending on cell
        else:
            print(f"[+] Executed cell {exec_count-1}/{total_code_cells}")

    # Write updated notebook with outputs
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    # Also make notebook available separately in workspace root
    with open(SEPARATE_NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[+] Notebook execution complete!")
    print(f"    Saved executed notebook: {NOTEBOOK_PATH}")
    print(f"    Saved standalone copy:   {SEPARATE_NOTEBOOK_PATH}")


if __name__ == "__main__":
    execute_notebook()
