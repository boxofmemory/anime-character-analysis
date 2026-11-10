from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

PATH_LIKE_SUFFIXES = ("_dir", "_path", "_yaml", "_file", "_root")
PATH_LIKE_KEYS = {
    "dataset_yaml",
    "base_model",
    "detector_model",
    "recognition_model",
    "tracker_yaml",
    "videos_dir",
    "input_video",
    "known_db_path",
    "stranger_db_path",
    "runtime_target_path",
    "project",
    "weights_dir",
    "output_dir",
    "tracker_configs",
}


def _is_path_like(key: str) -> bool:
    return key in PATH_LIKE_KEYS or key.endswith(PATH_LIKE_SUFFIXES)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_value(value: Any, base_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {child_key: _resolve_value(child_value, base_dir, child_key) for child_key, child_value in value.items()}
    if isinstance(value, list):
        if key and _is_path_like(key):
            return [_resolve_path(Path(item), base_dir) if isinstance(item, str) else item for item in value]
        return [_resolve_value(item, base_dir) for item in value]
    if isinstance(value, str) and key and _is_path_like(key):
        return _resolve_path(Path(value), base_dir)
    return value


def _serialize_yaml(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_yaml(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_serialize_yaml(child) for child in value]
    return value


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        content = handle.read()
    if yaml is not None:
        raw = yaml.safe_load(content) or {}
    else:
        raw = _fallback_yaml_load(content)
    return _resolve_value(raw, config_path.parent)


def dump_yaml(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        serializable = _serialize_yaml(payload)
        if yaml is not None:
            yaml.safe_dump(serializable, handle, allow_unicode=True, sort_keys=False)
        else:
            handle.write(_fallback_yaml_dump(serializable))


def _fallback_yaml_load(content: str) -> dict:
    lines = content.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, raw_line in enumerate(lines):
        line = _strip_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if text.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"Invalid YAML list item: {raw_line}")
            container.append(_parse_scalar(text[2:].strip()))
            continue

        key, _, value = text.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            next_kind = _infer_nested_kind(lines, index, indent)
            child: Any = [] if next_kind == "list" else {}
            container[key] = child
            stack.append((indent, child))
        else:
            container[key] = _parse_scalar(value)

    return root


def _infer_nested_kind(lines: list[str], start_index: int, current_indent: int) -> str:
    for raw_line in lines[start_index + 1 :]:
        line = _strip_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= current_indent:
            break
        return "list" if line.strip().startswith("- ") else "dict"
    return "dict"


def _parse_scalar(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index].rstrip()
    return line


def _fallback_yaml_dump(payload: Any, indent: int = 0) -> str:
    if isinstance(payload, dict):
        lines: list[str] = []
        for key, value in payload.items():
            prefix = " " * indent + f"{key}:"
            if isinstance(value, (dict, list)):
                lines.append(prefix)
                lines.append(_fallback_yaml_dump(value, indent + 2))
            else:
                lines.append(prefix + f" {_format_scalar(value)}")
        return "\n".join(lines) + ("\n" if indent == 0 else "")
    if isinstance(payload, list):
        lines = []
        for item in payload:
            prefix = " " * indent + "- "
            if isinstance(item, (dict, list)):
                lines.append(prefix.rstrip())
                lines.append(_fallback_yaml_dump(item, indent + 2))
            else:
                lines.append(prefix + _format_scalar(item))
        return "\n".join(lines)
    return " " * indent + _format_scalar(payload)


def _format_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)
