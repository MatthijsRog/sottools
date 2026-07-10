"""Fix stale ipython2 metadata in Jupyter notebooks."""

import json
from pathlib import Path

NOTEBOOKS = list(Path("docs/tutorials").glob("*.ipynb"))

for path in NOTEBOOKS:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["metadata"]["language_info"] = {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.8",
    }
    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Fixed: {path}")

for cell in nb.get("cells", []):
    for output in cell.get("outputs", []):
        output.pop("jetTransient", None)
