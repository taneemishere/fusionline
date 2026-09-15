#!/usr/bin/env python3
"""Export ToolRet to JSONL for the TypeScript harness.

Outputs (repo root):
    data/toolret/tools.jsonl  - {tool_id, name, description, search_text, documentation, category}
    data/toolret/tasks.jsonl  - {task_id, source, query, instruction, gt_ids, category}

Run from the repo root with requirements.txt installed:
    PYTHONPATH=. python3 scripts/export_toolret_jsonl.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolsel.data import load_tasks, load_tools


def main() -> None:
    tools = load_tools()
    tasks = load_tasks()

    tools_path = Path("data/toolret/tools.jsonl")
    tasks_path = Path("data/toolret/tasks.jsonl")

    with tools_path.open("w") as f:
        for t in tools.values():
            f.write(json.dumps(asdict(t)) + "\n")

    with tasks_path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t)) + "\n")

    print(f"wrote {len(tools)} tools -> {tools_path}")
    print(f"wrote {len(tasks)} tasks -> {tasks_path}")


if __name__ == "__main__":
    main()
