"""Flask API 路由。

把所有 NLP 能力、存储与流水线编排暴露为 REST 接口，
前端 10 个页面通过 ``fetch`` 调用这些接口。
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import uuid
from typing import Optional
from urllib.parse import quote

from flask import Blueprint, Response, current_app, jsonify, request

from nlp import (get_constituency_parser, get_embeddings, get_keywords, get_ner,
                 get_parser, get_segmenter, get_sentiment, get_summarizer,
                 get_tagger, get_translator, ENTITY_TYPE_NAMES, TAG_NAMES,
                 DEP_REL_NAMES, PHRASE_NAMES, POLARITY_NAMES)
from nlp.lexicon import STOPWORDS
from storage import StoreRegistry
from storage.corpus_service import (
    _doc_from_mapping,
    build_params,
    normalize_tags,
    parse_documents,
    prepare_document,
    query as query_corpus,
    serialize,
    tag_facets,
)


api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _registry() -> StoreRegistry:
    return current_app.config["STORE_REGISTRY"]


def _engine():
    return current_app.config["PIPELINE_ENGINE"]


def _models_dir() -> str:
    import os
    path = os.path.join(current_app.config["DATA_ROOT"], "models")
    os.makedirs(path, exist_ok=True)
    return path


def _store_result(task: str, text: str, result: dict,
                  corpus_id: Optional[str] = None) -> str:
    record = {"text": text, "result": result, "created_at": time.time()}
    if corpus_id:
        record["corpus_id"] = corpus_id
    return _registry().task(task).insert(record)


def _payload() -> dict:
    data = request.get_json(silent=True) or {}
    return data


def _resolve_text(data: dict) -> tuple[str, Optional[str]]:
    """从请求中取文本：优先 text，其次 corpus_id。"""
    if data.get("text"):
        return data["text"], data.get("corpus_id")
    corpus_id = data.get("corpus_id")
    if corpus_id:
        record = _registry().task("corpus").get(corpus_id)
        if record:
            return record.get("text", ""), corpus_id
        return "", corpus_id
    return "", None


def _clean(text: str, remove_stopwords: bool = True) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    seg = get_segmenter()
    words = seg.cut(text)
    if remove_stopwords:
        kept = [w for w in words if w not in STOPWORDS]
    else:
        kept = words
    removed = len(words) - len(kept)
    return {
        "text": text,
        "cleaned": " ".join(kept),
        "tokens": kept,
        "original_tokens": words,
        "removed_stopwords": removed,
    }


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@api.get("/status")
def status():
    return jsonify({
        "ok": True,
        "version": "1.0.0",
        "tasks": _registry().tasks(),
        "time": time.time(),
    })


@api.get("/meta")
def meta():
    """给前端提供标签集合与可配置参数。"""
    return jsonify({
        "tag_names": TAG_NAMES,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
        "entity_type_names": ENTITY_TYPE_NAMES,
        "polarity_names": POLARITY_NAMES,
        "directions": [{"id": "zh2en", "name": "中文 → 英文"},
                       {"id": "en2zh", "name": "英文 → 中文"}],
    })


# ---------------------------------------------------------------------------
# 语料库管理
# ---------------------------------------------------------------------------

def _list_corpus_legacy():
    records = _registry().task("corpus").all()
    items = [serialize(r) for r in records]
    items.sort(key=lambda x: (x.get("created_at", 0), x.get("id", "")), reverse=True)
    return jsonify({"corpora": items})


@api.get("/corpus")
def list_corpus():
    # 兼容旧页面：不带任何查询参数时返回原有结构
    if not request.args:
        return _list_corpus_legacy()
    try:
        params = build_params(request.args)
        result = query_corpus(_registry().task("corpus"), params)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    result.update({
        "filters": {
            "tags": params["tags"], "tag_mode": params["tag_mode"],
            "keyword": params["keyword"], "length_min": params["length_min"],
            "length_max": params["length_max"], "created_from": params["created_from"],
            "created_to": params["created_to"],
        },
        "facets": tag_facets(_registry().task("corpus"), params),
    })
    return jsonify(result)


@api.get("/corpus/options")
def corpus_options():
    items = [{"id": r.get("id"), "name": r.get("name", "未命名")}
             for r in _registry().task("corpus").all()]
    items.sort(key=lambda x: x.get("name", ""))
    return jsonify({"corpora": items})


@api.post("/corpus")
def create_corpus():
    data = _payload()
    try:
        default_tags = normalize_tags(data.get("tags"))
        record = prepare_document(data, default_tags=default_tags)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.post("/corpus/upload")
def upload_corpus():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "未接收到文件"}), 400
    try:
        default_tags = normalize_tags(request.form.getlist("tags"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    inserted = []
    errors = []
    store = _registry().task("corpus")
    for file in files:
        raw = file.read()
        filename = file.filename or "上传文件"
        try:
            documents, parse_errors = parse_documents(filename, raw)
        except ValueError as exc:
            documents, parse_errors = [], [{"name": filename, "error": str(exc)}]
        errors.extend(parse_errors)

        for doc in documents:
            try:
                record = prepare_document(
                    {"name": doc["name"], "text": doc["text"], "tags": doc["tags"]},
                    default_tags=default_tags,
                    fallback_name=filename)
                inserted.append({"id": store.insert(record), "name": record["name"],
                                 "length": len(record["text"]), "tags": record["tags"]})
            except (TypeError, ValueError, OSError) as exc:
                errors.append({"name": doc.get("name", filename), "error": str(exc)})
    return jsonify({
        "ok": bool(inserted),
        "inserted_count": len(inserted),
        "failed_count": len(errors),
        "inserted": inserted,
        "errors": errors,
    }), (200 if inserted else 400)


@api.post("/corpus/batch")
def batch_create_corpus():
    data = _payload()
    files = data.get("files")
    documents = data.get("documents")
    if not files and not documents:
        return jsonify({"error": "没有需要入库的文档"}), 400

    try:
        default_tags = normalize_tags(data.get("tags"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    parsed_docs = []
    errors = []
    if files:
        if not isinstance(files, list):
            return jsonify({"error": "files 必须是数组"}), 400
        for item in files:
            if not isinstance(item, dict) or "filename" not in item or "content" not in item:
                errors.append({"name": "请求项", "error": "文件项必须包含 filename 和 content"})
                continue
            try:
                raw = item["content"].encode("utf-8")
            except AttributeError:
                errors.append({"name": item.get("filename", "未知文件"), "error": "文件内容必须是字符串"})
                continue
            docs, file_errors = parse_documents(item["filename"], raw)
            parsed_docs.extend(docs)
            errors.extend(file_errors)
    if documents:
        if not isinstance(documents, list):
            return jsonify({"error": "documents 必须是数组"}), 400
        for index, item in enumerate(documents, 1):
            if not isinstance(item, dict):
                errors.append({"name": f"文档_{index}", "error": "文档必须是对象"})
                continue
            try:
                parsed_docs.append(_doc_from_mapping(item, f"文档_{index}"))
            except ValueError as exc:
                errors.append({"name": item.get("name", f"文档_{index}"), "error": str(exc)})

    inserted = []
    store = _registry().task("corpus")
    for doc in parsed_docs:
        try:
            record = prepare_document(doc, default_tags=default_tags)
            inserted.append({"id": store.insert(record), "name": record["name"],
                             "length": len(record["text"]), "tags": record.get("tags", [])})
        except (TypeError, ValueError, OSError) as exc:
            errors.append({"name": doc.get("name", "未知文档"), "error": str(exc)})

    return jsonify({
        "ok": bool(inserted),
        "inserted_count": len(inserted),
        "failed_count": len(errors),
        "inserted": inserted,
        "errors": errors,
    }), (200 if inserted else 400)


@api.get("/corpus/tags")
def corpus_tags():
    try:
        params = build_params(request.args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"tags": tag_facets(_registry().task("corpus"), params)})


@api.get("/corpus/export")
def export_corpus():
    try:
        params = build_params(request.args)
        fmt = (request.args.get("format") or "jsonl").lower()
        if fmt not in {"jsonl", "json", "csv", "txt"}:
            raise ValueError("导出格式只能是 jsonl、json、csv 或 txt")
        scope = (request.args.get("scope") or "all").lower()
        if scope not in {"all", "page"}:
            raise ValueError("导出范围只能是 all 或 page")
        result = query_corpus(_registry().task("corpus"), params, full=True,
                              page=(scope == "page"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    items = result["records"]

    if fmt == "jsonl":
        body = "\n".join(json.dumps(item, ensure_ascii=False) for item in items)
        mimetype = "application/x-ndjson; charset=utf-8"
        suffix = "jsonl"
    elif fmt == "json":
        body = json.dumps({"total": len(items), "documents": items},
                          ensure_ascii=False, indent=2)
        mimetype = "application/json; charset=utf-8"
        suffix = "json"
    elif fmt == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["id", "name", "tags", "length",
                                                    "created_at", "text"])
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["tags"] = ";".join(row.get("tags", []))
            writer.writerow(row)
        body = "﻿" + buffer.getvalue()
        mimetype = "text/csv; charset=utf-8"
        suffix = "csv"
    else:
        parts = []
        for item in items:
            parts.append(f"# {item['name']}\n# ID: {item['id']}\n"
                         f"# 标签: {';'.join(item.get('tags', []))}\n"
                         f"# 字数: {item['length']}\n\n{item['text']}\n")
        separator = "\n\n" + ("=" * 60) + "\n\n"
        body = separator.join(parts)
        mimetype = "text/plain; charset=utf-8"
        suffix = "txt"

    filename = f"corpus_export_{time.strftime('%Y%m%d_%H%M%S')}.{suffix}"
    response = Response(body, mimetype=mimetype)
    response.headers["Content-Disposition"] = (
        f"attachment; filename={filename}; filename*=UTF-8''{quote(filename)}")
    response.headers["X-Total-Count"] = str(len(items))
    return response


@api.get("/corpus/<cid>")
def get_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record or record.get("_deleted"):
        return jsonify({"error": "语料不存在"}), 404
    return jsonify(record)


@api.put("/corpus/<cid>/tags")
def update_corpus_tags(cid: str):
    data = _payload()
    try:
        tags = normalize_tags(data.get("tags"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    record = _registry().task("corpus").update(cid, {"tags": tags})
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    return jsonify({"ok": True, "id": cid, "tags": tags})


@api.delete("/corpus/<cid>")
def delete_corpus(cid: str):
    ok = _registry().task("corpus").delete(cid)
    return jsonify({"ok": ok})


@api.post("/corpus/<cid>/clean")
def clean_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record or record.get("_deleted"):
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    result = _clean(record.get("text", ""), data.get("remove_stopwords", True))
    _store_result("clean", record.get("text", ""), result, corpus_id=cid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 分词与词性标注
# ---------------------------------------------------------------------------

@api.post("/segment")
def segment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    seg = get_segmenter()
    words = seg.cut(text)
    result = {"words": words, "count": len(words)}
    rid = _store_result("segment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/pos")
def pos_tag():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    tagger = get_tagger()
    tokens = [[w, t] for w, t in tagger.tag(text)]
    result = {"tokens": tokens, "tag_names": TAG_NAMES}
    rid = _store_result("pos", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 句法分析
# ---------------------------------------------------------------------------

@api.post("/parse")
def parse():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    result = {
        "dependency": dep,
        "constituency": const,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
    }
    rid = _store_result("parse", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 命名实体识别与标注
# ---------------------------------------------------------------------------

@api.post("/ner")
def ner():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    entities = get_ner().recognize(text)
    result = {"entities": entities, "entity_type_names": ENTITY_TYPE_NAMES}
    rid = _store_result("ner", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/ner/annotate")
def ner_annotate():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    record = {
        "text": text,
        "entities": data.get("entities", []),
        "note": data.get("note", ""),
        "created_at": time.time(),
    }
    rid = _registry().task("annotation").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.get("/ner/annotations")
def ner_annotations():
    records = _registry().task("annotation").all()
    return jsonify({"annotations": records})


# ---------------------------------------------------------------------------
# 情感分析
# ---------------------------------------------------------------------------

@api.post("/sentiment")
def sentiment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_sentiment().analyze(text)
    result["polarity_name"] = POLARITY_NAMES.get(result["polarity"], "")
    rid = _store_result("sentiment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------

@api.post("/summary")
def summary():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_summarizer().summarize(
        text, ratio=data.get("ratio", 0.3),
        max_sentences=data.get("max_sentences"))
    rid = _store_result("summary", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 机器翻译（模拟）
# ---------------------------------------------------------------------------

@api.post("/translate")
def translate():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_translator().translate(text, direction=data.get("direction", "zh2en"))
    rid = _store_result("translate", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

@api.post("/keywords")
def keywords():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_keywords().extract(text, top_k=data.get("top_k", 10),
                                    method=data.get("method", "hybrid"))
    rid = _store_result("keywords", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 词向量
# ---------------------------------------------------------------------------

def _embedding_path() -> str:
    import os
    return os.path.join(_models_dir(), "embeddings.json")


@api.post("/embeddings/train")
def train_embeddings():
    data = _payload()
    corpus_ids = data.get("corpus_ids")
    store = _registry().task("corpus")
    if corpus_ids:
        texts = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
    else:
        texts = [r["text"] for r in store.all() if not r.get("_deleted")]
    if not texts:
        return jsonify({"error": "没有可用语料，请先上传语料"}), 400

    emb = get_embeddings()
    emb.train(texts, vocab_size=data.get("vocab_size", 200),
              dim=data.get("dim", 20), window=data.get("window", 5),
              min_count=data.get("min_count", 1))

    payload = {
        "vocab": emb.vocab,
        "vectors": emb.vectors,
        "dim": emb.dim,
        "trained_at": time.time(),
        "corpus_count": len(texts),
    }
    with open(_embedding_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return jsonify(emb.stats())


@api.get("/embeddings/vectors")
def embeddings_vectors():
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    if not emb.vectors:
        return jsonify({"error": "尚未训练词向量"}), 404
    n_clusters = int(request.args.get("clusters", 5))
    proj = emb.project_2d()
    clusters = emb.cluster(n_clusters)
    return jsonify({
        "points": [{"word": w, "x": round(p[0], 4), "y": round(p[1], 4),
                    "cluster": clusters.get(w, 0)} for w, p in proj.items()],
        "stats": emb.stats(),
    })


@api.get("/embeddings/neighbors")
def embeddings_neighbors():
    word = request.args.get("word", "")
    k = int(request.args.get("k", 10))
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    return jsonify({"word": word, "neighbors": emb.nearest(word, k)})


def _load_embeddings():
    import os
    path = _embedding_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        emb = get_embeddings()
        emb.vocab = data.get("vocab", [])
        emb.vectors = data.get("vectors", {})
        emb.dim = data.get("dim", 0)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 流水线配置与执行
# ---------------------------------------------------------------------------

@api.get("/pipeline/stages")
def pipeline_stages():
    return jsonify({"stages": _engine().list_stages()})


@api.post("/pipeline")
def save_pipeline():
    data = _payload()
    config = data.get("config") or data
    if not config.get("stages"):
        return jsonify({"error": "流水线至少需要一个阶段"}), 400
    name = config.get("name") or f"流水线_{int(time.time())}"
    record = {"name": name, "config": config, "created_at": time.time()}
    rid = _registry().task("pipeline_config").insert(record)
    return jsonify({"id": rid, "name": name, "ok": True})


@api.get("/pipeline")
def list_pipelines():
    records = _registry().task("pipeline_config").all()
    items = [{"id": r["id"], "name": r.get("name"), "config": r.get("config"),
              "created_at": r.get("created_at")}
             for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"pipelines": items})


@api.get("/pipeline/<pid>")
def get_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    return jsonify(record)


@api.post("/pipeline/preview")
def pipeline_preview():
    """对单条文本跑流水线（不持久化），供配置页预览。"""
    data = _payload()
    text = (data.get("text") or "").strip()
    config = data.get("config")
    if not text or not config:
        return jsonify({"error": "缺少文本或配置"}), 400
    try:
        result = _engine().build(config).run({"text": text})
        return jsonify({"ok": True, "output": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@api.post("/pipeline/<pid>/run")
def run_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    config = record.get("config")
    data = _payload()

    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if data.get("batch"):
        # 批量：对语料库中的多篇文档执行
        corpus_ids = data.get("corpus_ids") or []
        store = _registry().task("corpus")
        docs = []
        if corpus_ids:
            docs = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
        else:
            docs = [r["text"] for r in store.all() if not r.get("_deleted")]
        if not docs:
            return jsonify({"error": "没有可处理的文档"}), 400

        progress_state = {"done": 0, "total": len(docs)}

        def _progress(done, total):
            progress_state["done"] = done
            progress_state["total"] = total

        results = _engine().run_batch(
            config, docs, shared=data.get("shared"),
            max_workers=data.get("max_workers", 4),
            chunk_size=data.get("chunk_size", 16),
            progress=_progress)
        succeeded = sum(1 for r in results if r and r["ok"])
        failed = len(results) - succeeded
        run_record = {
            "run_id": run_id, "pipeline_id": pid, "batch": True,
            "doc_count": len(docs), "succeeded": succeeded, "failed": failed,
            "started": started, "finished": time.time(),
            "results": results,
        }
        rid = _registry().task("pipeline_run").insert(run_record)
        return jsonify({"run_id": run_id, "id": rid, "succeeded": succeeded,
                        "failed": failed, "doc_count": len(docs)})
    else:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "缺少文本"}), 400
        try:
            output = _engine().build(config).run({"text": text})
            run_record = {
                "run_id": run_id, "pipeline_id": pid, "batch": False,
                "text": text, "output": output,
                "started": started, "finished": time.time(),
            }
            rid = _registry().task("pipeline_run").insert(run_record)
            return jsonify({"run_id": run_id, "id": rid, "ok": True,
                            "output": output})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400


@api.get("/pipeline/run/<run_id>")
def get_pipeline_run(run_id: str):
    records = _registry().task("pipeline_run").query(
        where=[("run_id", "eq", run_id)])
    if not records:
        return jsonify({"error": "执行记录不存在"}), 404
    return jsonify(records[0])


# ---------------------------------------------------------------------------
# 结果查询（分片合并与查询）
# ---------------------------------------------------------------------------

@api.get("/results")
def list_result_tasks():
    registry = _registry()
    tasks = []
    for name in registry.tasks():
        if name in ("corpus", "pipeline_config", "annotation"):
            continue
        stats = registry.task(name).stats()
        tasks.append(stats)
    return jsonify({"tasks": tasks})


@api.get("/results/<task>")
def query_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    store = registry.task(task)
    where = []
    for key in ("type", "corpus_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    order_by = request.args.get("order_by")
    order = request.args.get("order", "desc")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)
    records = store.query(where=where or None, order_by=order_by,
                          order=order, limit=limit, offset=offset)
    return jsonify({
        "task": task,
        "count": len(records),
        "stats": store.stats(),
        "records": records,
    })


@api.post("/results/<task>/compact")
def compact_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).compact())


@api.get("/results/<task>/merge")
def merge_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).merge())


@api.post("/results/compact_all")
def compact_all():
    return jsonify({"compacted": _registry().compact_all()})
