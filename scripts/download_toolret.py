#!/usr/bin/env python3
"""Download the public ToolRet datasets from HuggingFace.

Run from the repo root:
    python scripts/download_toolret.py

Writes to:
    data/toolret/tools/{code,customized,web}
    data/toolret/queries/{code,customized,web}
"""

from pathlib import Path

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError as exc:  # pragma: no cover
    raise ImportError("'datasets' is not installed. Run: pip install -r requirements.txt") from exc

DATA_DIR = Path(__file__).parent.parent / "data" / "toolret"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for name, repo_id in [("tools", "mangopy/ToolRet-Tools"),
                          ("queries", "mangopy/ToolRet-Queries")]:
        for cfg in get_dataset_config_names(repo_id):
            print(f"Loading {repo_id} / {cfg} ...")
            out = DATA_DIR / name / cfg
            load_dataset(repo_id, cfg).save_to_disk(out)
            print(f"Saved {repo_id}/{cfg} -> {out}")
    print("Done. ToolRet data is in:", DATA_DIR)


if __name__ == "__main__":
    main()
