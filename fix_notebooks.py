"""Re-execute the docs notebooks with clean plots, then strip PyCharm metadata.

Run from the repository root, inside the sottools environment:
    python tools/refresh_notebooks.py            # execute + clean all notebooks
    python tools/refresh_notebooks.py --no-run   # only clean metadata
"""

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK_DIR = Path("docs/tutorials")
EXCLUDE = ("DELETEME",)  # notebooks whose name contains any of these are skipped
TIMEOUT = None  # no per-cell timeout; the inversion notebooks can take a while

LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.12.8",
}
KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}

run = "--no-run" not in sys.argv

# Clean environment for the kernels, independent of PyCharm and of your own settings:
# - an empty matplotlib config dir, so no user matplotlibrc or style is picked up;
# - a fresh IPython profile whose only setting makes inline figures opaque white.
#   (The inline backend's default facecolor is transparent, which is what renders
#   dark on a dark IDE or docs theme.)
# A fixed directory is reused so matplotlib's font cache is only built once.
clean_dir = Path(tempfile.gettempdir()) / "sottools-nb-refresh"
(clean_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
profile = clean_dir / "ipython" / "profile_default"
profile.mkdir(parents=True, exist_ok=True)
(profile / "ipython_kernel_config.py").write_text(
    "c.InlineBackend.rc = {\n"
    "    'figure.facecolor': 'white',\n"
    "    'figure.edgecolor': 'white',\n"
    "    'savefig.facecolor': 'white',\n"
    "}\n",
    encoding="utf-8",
)
os.environ["MPLCONFIGDIR"] = str(clean_dir / "matplotlib")
os.environ["IPYTHONDIR"] = str(clean_dir / "ipython")
os.environ.pop("MPLBACKEND", None)

notebooks = sorted(
    p for p in NOTEBOOK_DIR.glob("*.ipynb") if not any(x in p.name for x in EXCLUDE)
)
print(f"Python: {sys.executable}")

for path in notebooks:
    nb = nbformat.read(path, as_version=4)

    if run:
        print(f"Running: {path} ...", flush=True)
        client = NotebookClient(
            nb,
            kernel_name="python3",
            timeout=TIMEOUT,
            record_timing=False,  # no execution timestamps in the diff
            resources={"metadata": {"path": str(path.parent)}},  # run in the
            # notebook's folder
        )
        try:
            client.execute()
        except CellExecutionError as err:
            print(f"FAILED: {path} (file left untouched)\n{err}")
            sys.exit(1)

    # Strip PyCharm metadata
    nb.metadata["kernelspec"] = KERNELSPEC
    nb.metadata["language_info"] = LANGUAGE_INFO
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            output.pop("jetTransient", None)

    nbformat.write(nb, path)
    print(f"Done: {path}")
