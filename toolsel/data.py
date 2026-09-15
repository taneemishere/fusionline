"""Load ToolRet tools and queries into a uniform Task / ToolRecord contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetDict, load_from_disk

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "toolret"

NAME_KEYS = ("name", "tool_name", "api_name", "function_name", "title")
DESC_KEYS = ("description", "desc", "functionality", "description_for_human",
             "summary", "purpose")
PARAM_KEYS = ("parameters", "doc_arguments", "required_parameters",
              "optional_parameters", "api_arguments", "arguments",
              "input_schema", "args", "params")


@dataclass(frozen=True)
class ToolRecord:
    tool_id: str
    name: str
    description: str
    search_text: str
    documentation: str
    category: str


@dataclass(frozen=True)
class Task:
    task_id: str
    source: str
    query: str
    instruction: str
    gt_ids: tuple[str, ...]
    category: str


def _parse_doc(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _pick(d: dict, keys) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _first_split(ds):
    if isinstance(ds, DatasetDict):
        return next(iter(ds.values()))
    return ds


def _parse_labels(raw) -> list[dict]:
    v = raw
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    return [x for x in v if isinstance(x, dict) and x.get("id")]


_TURN_RE = re.compile(
    r"(?:^|\n)\s*(?:user|human)\s*:\s*(.*?)(?=\n\s*(?:user|human|assist(?:ant)?|bot|system)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def clean_query(q: str) -> str:
    """Strip embedded `user:`/`assist:` transcript markup from query strings."""
    if not isinstance(q, str):
        return ""
    if re.search(r"(?:^|\n)\s*(?:user|human|assist(?:ant)?|system)\s*:", q, re.IGNORECASE):
        parts = [m.group(1).strip() for m in _TURN_RE.finditer(q)]
        parts = [p for p in parts if p]
        if parts:
            return " ".join(parts)
    return q.strip()


def load_tools(root: Path | str = DATA_DIR / "tools",
               subsets: list[str] | None = None) -> dict[str, ToolRecord]:
    root = Path(root)
    out: dict[str, ToolRecord] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or (subsets and sub.name not in subsets):
            continue
        d = _first_split(load_from_disk(sub))
        for row in d:
            doc = _parse_doc(row["documentation"])
            name = _pick(doc, NAME_KEYS) or row["id"]
            desc = _pick(doc, DESC_KEYS)
            params = " ".join(
                json.dumps(doc[k], default=str) for k in PARAM_KEYS if k in doc
            )
            search_text = f"{name}\n{desc}\n{params}".strip()
            out[row["id"]] = ToolRecord(
                tool_id=row["id"],
                name=name,
                description=desc,
                search_text=search_text,
                documentation=row["documentation"],
                category=sub.name,
            )
    return out


def load_tasks(root: Path | str = DATA_DIR / "queries",
               configs: list[str] | None = None,
               limit_per_config: int | None = None) -> list[Task]:
    root = Path(root)
    out: list[Task] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or (configs and sub.name not in configs):
            continue
        d = _first_split(load_from_disk(sub))
        n = len(d) if limit_per_config is None else min(limit_per_config, len(d))
        for row in d.select(range(n)):
            gt_ids = tuple(x["id"] for x in _parse_labels(row["labels"]))
            if not gt_ids:
                continue
            out.append(Task(
                task_id=row["id"],
                source=sub.name,
                query=clean_query(row["query"]),
                instruction=row.get("instruction", "") or "",
                gt_ids=gt_ids,
                category=row["category"],
            ))
    return out
