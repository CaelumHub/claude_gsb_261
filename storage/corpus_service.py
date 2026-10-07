"""语料库的批量解析、标签与跨分片查询服务。"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import time
from typing import Any, Iterable, Optional

from storage import ShardedStore

TEXT_KEYS = ("text", "content", "body", "正文", "内容", "文本")
NAME_KEYS = ("name", "title", "标题", "名称", "filename")
TAG_KEYS = ("tags", "labels", "标签")
SORT_FIELDS = {"created_at", "length", "name"}
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16")
_TAG_SPLIT_RE = re.compile(r"[,，、;；\n\r\t]+")


# ---------------------------------------------------------------------------
# 标签与文档规范化
# ---------------------------------------------------------------------------

def normalize_tags(value: Any = None, *, max_tags: int = 30) -> list[str]:
    """把字符串、列表等输入规范成去重后的标签列表。"""
    if value is None:
        raw_values: list[Any] = []
    elif isinstance(value, str):
        raw_values = _TAG_SPLIT_RE.split(value)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        raw_values = []
        for item in value:
            raw_values.extend(_TAG_SPLIT_RE.split(str(item)))
    else:
        raise ValueError("标签必须是字符串或字符串数组")

    tags: list[str] = []
    seen = set()
    for item in raw_values:
        tag = str(item).strip()
        if not tag:
            continue
        if len(tag) > 32:
            raise ValueError(f"标签过长：{tag[:32]}…")
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    if len(tags) > max_tags:
        raise ValueError(f"每篇文档最多设置 {max_tags} 个标签")
    return tags


def decode_bytes(raw: bytes) -> str:
    """按常见中文编码尝试解码上传文件。"""
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("无法识别文件编码，请保存为 UTF-8 后重试")


def _pick_value(data: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _doc_from_mapping(data: dict, fallback_name: Optional[str] = None) -> dict:
    text = _pick_value(data, TEXT_KEYS)
    if text is None:
        raise ValueError("缺少 text 字段")
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        raise ValueError("文档内容为空")

    name = _pick_value(data, NAME_KEYS)
    name = str(name).strip() if name is not None else ""
    tags = normalize_tags(_pick_value(data, TAG_KEYS))
    return {
        "name": name or fallback_name or "",
        "text": text,
        "tags": tags,
    }


def parse_documents(filename: str, raw: bytes) -> tuple[list[dict], list[dict]]:
    """解析单个上传文件为多篇文档。

    返回 ``(documents, errors)``。某个文件内的个别 JSON/JSONL 文档异常只会
    记录到 errors，不会影响同文件中的其它文档。
    """
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in {".txt", ".md", ".json", ".jsonl", ".csv"}:
        return [], [{"name": filename, "error": "仅支持 .txt/.md/.json/.jsonl/.csv 文件"}]

    try:
        content = decode_bytes(raw)
    except ValueError as exc:
        return [], [{"name": filename, "error": str(exc)}]

    fallback = os.path.splitext(os.path.basename(filename or ""))[0]
    if suffix in {".txt", ".md"}:
        text = content.strip()
        if not text:
            return [], [{"name": filename, "error": "文档内容为空"}]
        return [{"name": fallback, "text": text, "tags": []}], []

    if suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return [], [{"name": filename, "error": f"JSON 格式错误：{exc.msg}"}]
        if isinstance(data, dict):
            for container in ("documents", "docs", "items", "corpus"):
                if isinstance(data.get(container), list):
                    data = data[container]
                    break
            else:
                try:
                    doc = _doc_from_mapping(data, fallback)
                    return [doc], []
                except ValueError as exc:
                    return [], [{"name": filename, "error": str(exc)}]
        if not isinstance(data, list):
            return [], [{"name": filename, "error": "JSON 顶层必须是文档对象或文档数组"}]
        return _docs_from_sequence(data, fallback)

    if suffix == ".jsonl":
        docs = []
        errors = []
        for line_no, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("每行必须是 JSON 对象")
                docs.append(_doc_from_mapping(data, f"{fallback}_{line_no}"))
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append({"name": f"{filename}:{line_no}", "error": str(exc)})
        if not docs and not errors:
            errors.append({"name": filename, "error": "文档内容为空"})
        return docs, errors

    # CSV
    return _parse_csv(content, filename, fallback)


def _docs_from_sequence(items: Iterable[Any], fallback: str) -> tuple[list[dict], list[dict]]:
    docs = []
    errors = []
    for index, item in enumerate(items, 1):
        try:
            if not isinstance(item, dict):
                raise ValueError("文档必须是对象")
            docs.append(_doc_from_mapping(item, f"{fallback}_{index}"))
        except ValueError as exc:
            errors.append({"name": f"{fallback}_{index}", "error": str(exc)})
    return docs, errors


def _parse_csv(content: str, filename: str, fallback: str) -> tuple[list[dict], list[dict]]:
    try:
        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            return [], [{"name": filename, "error": "CSV 缺少表头"}]
        header = {h.strip().lower(): h.strip() for h in reader.fieldnames if h}
        text_keys = {x.lower() for x in TEXT_KEYS}
        name_keys = {x.lower() for x in NAME_KEYS}
        tag_keys = {x.lower() for x in TAG_KEYS}
        text_key = next((h for key, h in header.items() if key in text_keys), None)
        if not text_key:
            return [], [{"name": filename, "error": "CSV 必须包含 text/content/正文 列"}]
        name_key = next((h for key, h in header.items() if key in name_keys), None)
        tag_key = next((h for key, h in header.items() if key in tag_keys), None)

        docs = []
        errors = []
        for row_no, row in enumerate(reader, 2):
            try:
                data = {"text": row.get(text_key)}
                if name_key:
                    data["name"] = row.get(name_key)
                if tag_key:
                    data["tags"] = row.get(tag_key)
                docs.append(_doc_from_mapping(data, f"{fallback}_{row_no}"))
            except ValueError as exc:
                errors.append({"name": f"{filename}:{row_no}", "error": str(exc)})
        return docs, errors
    except csv.Error as exc:
        return [], [{"name": filename, "error": f"CSV 解析失败：{exc}"}]


def prepare_document(data: dict, default_tags: Optional[list[str]] = None,
                     fallback_name: Optional[str] = None) -> dict:
    doc = _doc_from_mapping(data, fallback_name)
    tags = normalize_tags(list(default_tags or []) + doc["tags"])
    now = time.time()
    return {
        "name": doc["name"] or f"文档_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}",
        "text": doc["text"],
        "tags": tags,
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# 序列化与筛选
# ---------------------------------------------------------------------------

def serialize(record: dict, *, full: bool = False) -> dict:
    text = record.get("text", "")
    item = {
        "id": record.get("id"),
        "name": record.get("name", "未命名"),
        "tags": list(record.get("tags") or []),
        "length": len(text),
        "created_at": record.get("created_at") or record.get("_created") or 0,
        "preview": text[:160],
    }
    if full:
        item["text"] = text
    return item


def _parse_time(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    value = str(value).strip()
    if value.isdigit():
        number = float(value)
        return number / 1000 if number > 10_000_000_000 else number
    try:
        from datetime import datetime
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        raise ValueError(f"无法识别的时间：{value}")


def _arg_int(args, name: str, default: Optional[int] = None,
             minimum: Optional[int] = None, maximum: Optional[int] = None) -> Optional[int]:
    raw = args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} 必须是整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} 不能大于 {maximum}")
    return value


def build_params(args) -> dict:
    """从 Flask query args 构造规范化查询条件。"""
    tag_values = args.getlist("tags") or args.getlist("tag")
    tags = normalize_tags(t for value in tag_values for t in _TAG_SPLIT_RE.split(value))
    keyword = (args.get("q") or args.get("keyword") or "").strip()
    sort = args.get("sort", "created_at")
    if sort not in SORT_FIELDS:
        raise ValueError("排序字段只能是 created_at、length、name")
    order = args.get("order", "desc").lower()
    if order not in {"asc", "desc"}:
        raise ValueError("排序方向只能是 asc 或 desc")
    tag_mode = args.get("tag_mode", "any").lower()
    if tag_mode not in {"any", "all"}:
        raise ValueError("标签匹配方式只能是 any 或 all")
    page_size = _arg_int(args, "page_size", _arg_int(args, "limit", 20), 1, 200)
    params = {
        "tags": tags,
        "tag_mode": tag_mode,
        "keyword": keyword,
        "length_min": _arg_int(args, "length_min", None, 0),
        "length_max": _arg_int(args, "length_max", None, 0),
        "created_from": _parse_time(args.get("created_from")),
        "created_to": _parse_time(args.get("created_to")),
        "sort": sort,
        "order": order,
        "page_size": page_size,
        "cursor": args.get("cursor") or None,
        "direction": args.get("direction", "next").lower(),
    }
    if (params["length_min"] is not None and params["length_max"] is not None
            and params["length_min"] > params["length_max"]):
        raise ValueError("最短长度不能大于最长长度")
    if (params["created_from"] is not None and params["created_to"] is not None
            and params["created_from"] > params["created_to"]):
        raise ValueError("开始时间不能晚于结束时间")
    if params["direction"] not in {"next", "prev"}:
        raise ValueError("翻页方向只能是 next 或 prev")
    return params


def _base_matches(record: dict, params: dict) -> bool:
    text = record.get("text", "") or ""
    name = record.get("name", "") or ""
    length = len(text)
    created_at = record.get("created_at") or record.get("_created") or 0

    if params["length_min"] is not None and length < params["length_min"]:
        return False
    if params["length_max"] is not None and length > params["length_max"]:
        return False
    if params["created_from"] is not None and created_at < params["created_from"]:
        return False
    if params["created_to"] is not None and created_at > params["created_to"]:
        return False
    if params["keyword"]:
        needle = params["keyword"].casefold()
        haystack = f"{name}\n{text}".casefold()
        if needle not in haystack:
            return False

    tags = set(record.get("tags") or [])
    wanted = set(params["tags"])
    if wanted:
        if params["tag_mode"] == "all" and not wanted.issubset(tags):
            return False
        if params["tag_mode"] == "any" and not (wanted & tags):
            return False
    return True


def _sort_key(record: dict, field: str):
    if field == "length":
        text = record.get("text", "") or ""
        value = len(text) if isinstance(text, str) else len(str(text))
    elif field == "name":
        value = str(record.get("name") or "").casefold()
    else:
        raw = record.get("created_at") or record.get("_created") or 0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
    return value, str(record.get("id") or "")


def _filter_signature(params: dict) -> str:
    payload = {
        "tags": params["tags"],
        "tag_mode": params["tag_mode"],
        "keyword": params["keyword"],
        "length_min": params["length_min"],
        "length_max": params["length_max"],
        "created_from": params["created_from"],
        "created_to": params["created_to"],
        "sort": params["sort"],
        "order": params["order"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def encode_cursor(record: dict, params: dict) -> str:
    payload = {
        "v": _sort_key(record, params["sort"])[0],
        "id": record.get("id"),
        "f": params["sort"],
        "o": params["order"],
        "h": _filter_signature(params),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, params: dict) -> tuple[Any, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if payload.get("f") != params["sort"] or payload.get("o") != params["order"]:
            raise ValueError("游标与当前排序不一致")
        if payload.get("h") != _filter_signature(params):
            raise ValueError("筛选条件已变化，请返回第一页")
        return payload.get("v"), str(payload.get("id"))
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"无效的分页游标：{exc}")


def _is_before_in_order(record_key: tuple, cursor_value: Any, cursor_id: str,
                        descending: bool) -> bool:
    """记录在当前排序方向中是否位于游标之前。"""
    if descending:
        return (record_key[0] > cursor_value
                or (record_key[0] == cursor_value and record_key[1] > cursor_id))
    return (record_key[0] < cursor_value
            or (record_key[0] == cursor_value and record_key[1] < cursor_id))


def _is_after_in_order(record_key: tuple, cursor_value: Any, cursor_id: str,
                       descending: bool) -> bool:
    """记录在当前排序方向中是否位于游标之后。"""
    if descending:
        return (record_key[0] < cursor_value
                or (record_key[0] == cursor_value and record_key[1] < cursor_id))
    return (record_key[0] > cursor_value
            or (record_key[0] == cursor_value and record_key[1] > cursor_id))


def query(store: ShardedStore, params: dict, *, full: bool = False,
          page: bool = True) -> dict:
    """跨全部分片筛选、全局稳定排序，并进行游标分页。"""
    records = [r for r in store.all() if not r.get("_deleted") and _base_matches(r, params)]
    records.sort(key=lambda r: _sort_key(r, params["sort"]),
                 reverse=params["order"] == "desc")
    total = len(records)

    if not page:
        return {
            "total": total,
            "records": [serialize(r, full=full) for r in records],
            "next_cursor": None,
            "prev_cursor": None,
            "page_size": total,
        }

    size = params["page_size"]
    next_cursor = None
    prev_cursor = None
    if params["cursor"]:
        value, cursor_id = decode_cursor(params["cursor"], params)
        descending = params["order"] == "desc"
        if params["direction"] == "next":
            after = []
            before_count = 0
            for record in records:
                key = _sort_key(record, params["sort"])
                if _is_after_in_order(key, value, cursor_id, descending):
                    after.append(record)
                else:
                    before_count += 1
            has_next = len(after) > size
            page_records = after[:size]
            if page_records and before_count:
                prev_cursor = encode_cursor(page_records[0], params)
            if page_records and has_next:
                next_cursor = encode_cursor(page_records[-1], params)
        else:
            before = []
            has_next = False
            for record in records:
                key = _sort_key(record, params["sort"])
                if _is_before_in_order(key, value, cursor_id, descending):
                    before.append(record)
                else:
                    has_next = True
            has_prev = len(before) > size
            page_records = before[-size:]
            if page_records and has_prev:
                prev_cursor = encode_cursor(page_records[0], params)
            if page_records and has_next:
                next_cursor = encode_cursor(page_records[-1], params)
    else:
        has_next = len(records) > size
        page_records = records[:size]
        if page_records and has_next:
            next_cursor = encode_cursor(page_records[-1], params)

    return {
        "total": total,
        "records": [serialize(r, full=full) for r in page_records],
        "next_cursor": next_cursor,
        "prev_cursor": prev_cursor,
        "page_size": size,
    }


def tag_facets(store: ShardedStore, params: dict) -> list[dict]:
    """统计其它筛选条件下各标签的文档数（标签条件本身不参与收窄）。"""
    facet_params = dict(params)
    facet_params["tags"] = []
    counts: dict[str, int] = {}
    for record in store.all():
        if record.get("_deleted") or not _base_matches(record, facet_params):
            continue
        for tag in record.get("tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
    return [{"name": name, "count": count}
            for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
