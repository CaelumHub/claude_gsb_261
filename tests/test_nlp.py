"""NLP 平台单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：分词、词性、句法、NER、情感、摘要、翻译、关键词、词向量、
分片存储（含并发锁）、流水线引擎、HMM。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import (get_segmenter, get_tagger, get_parser, get_constituency_parser,
                 get_ner, get_sentiment, get_summarizer, get_translator,
                 get_keywords, get_embeddings, TAGSET)
from nlp.hmm import HMM
from pipeline import PipelineEngine, PipelineError
from storage import ShardedStore, StoreRegistry
from storage.corpus_service import query as query_corpus

try:
    from app import create_app
except ImportError:  # Flask 未安装时，纯存储/NLP 测试仍可运行
    create_app = None


class TestSegmenter(unittest.TestCase):
    def test_basic(self):
        words = get_segmenter().cut("自然语言处理是人工智能的重要分支")
        self.assertIn("自然语言", words)
        self.assertIn("人工智能", words)
        self.assertIn("是", words)

    def test_english_number(self):
        words = get_segmenter().cut("我用Python写了100行代码")
        self.assertIn("Python", words)
        self.assertIn("100", words)


class TestPOSTagger(unittest.TestCase):
    def test_tags(self):
        pairs = get_tagger().tag("我学习自然语言处理")
        self.assertTrue(pairs)
        for word, tag in pairs:
            self.assertIn(tag, TAGSET, f"{word}:{tag}")

    def test_punct_as_other(self):
        pairs = get_tagger().tag("你好，世界。")
        tags = [t for _, t in pairs]
        for t in tags:
            if t in ("，", "。"):
                continue
        # 标点词本身应为 x
        for w, t in pairs:
            if w in ("，", "。"):
                self.assertEqual(t, "x")


class TestParser(unittest.TestCase):
    def test_dependency(self):
        dep = get_parser().parse("北京大学的研究团队开发了机器学习系统")
        self.assertEqual(len(dep["words"]), len(dep["heads"]))
        self.assertIn(-1, dep["heads"])  # 存在根
        # 每个 head 都是有效下标或 -1
        for h in dep["heads"]:
            self.assertTrue(h == -1 or 0 <= h < len(dep["words"]))

    def test_constituency_spans(self):
        c = get_constituency_parser().parse("北京大学的研究团队开发了系统")
        leaves = self._leaves(c["tree"])
        self.assertEqual("北京大学的研究团队开发了系统", leaves)

    @staticmethod
    def _leaves(tree):
        if not tree.get("children"):
            return tree.get("word", "")
        return "".join(TestParser._leaves(ch) for ch in tree["children"])


class TestNER(unittest.TestCase):
    def test_known_entities(self):
        ents = get_ner().recognize("马云在北京工作")
        types = {e["text"]: e["type"] for e in ents}
        self.assertEqual(types.get("马云"), "PERSON")
        self.assertEqual(types.get("北京"), "LOCATION")

    def test_date_money(self):
        ents = get_ner().recognize("2024年10月1日花了99.9元")
        texts = [e["text"] for e in ents]
        self.assertTrue(any("2024" in t for t in texts))
        self.assertTrue(any("99.9" in t for t in texts))


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        r = get_sentiment().analyze("这个产品非常好用，我很喜欢")
        self.assertEqual(r["polarity"], "positive")

    def test_negative(self):
        r = get_sentiment().analyze("服务态度很差，令人失望")
        self.assertEqual(r["polarity"], "negative")


class TestSummarizer(unittest.TestCase):
    def test_shorter(self):
        text = ("自然语言处理是人工智能的重要分支。它研究如何让计算机理解语言。"
                "分词是基础任务。词性标注是另一个任务。")
        r = get_summarizer().summarize(text, ratio=0.5)
        self.assertTrue(len(r["summary"]) < len(text))
        self.assertTrue(r["top_indices"])


class TestTranslator(unittest.TestCase):
    def test_zh2en(self):
        r = get_translator().translate("我喜欢机器学习", "zh2en")
        self.assertIn("machine learning", r["translation"].lower())

    def test_en2zh(self):
        r = get_translator().translate("I like China", "en2zh")
        self.assertTrue(r["translation"])


class TestKeywords(unittest.TestCase):
    def test_extract(self):
        r = get_keywords().extract("自然语言处理是人工智能的重要分支", top_k=5)
        self.assertTrue(r["keywords"])
        for k in r["keywords"]:
            self.assertIn("word", k)
            self.assertIn("score", k)


class TestEmbeddings(unittest.TestCase):
    def test_train_nearest(self):
        texts = [
            "自然语言处理是人工智能的重要分支",
            "机器学习是人工智能的核心技术",
            "深度学习推动了人工智能的发展",
            "分词是自然语言处理的基础任务",
        ] * 3
        emb = get_embeddings()
        emb.train(texts, vocab_size=60, dim=8, window=3, min_count=1)
        self.assertTrue(emb.vocab)
        self.assertTrue(emb.vectors)
        # 近邻应返回词且不包含自身
        nb = emb.nearest(emb.vocab[0], k=3)
        self.assertTrue(nb)
        self.assertNotIn(emb.vocab[0], [n["word"] for n in nb])
        # 2D 投影
        proj = emb.project_2d()
        self.assertEqual(len(proj), len(emb.vectors))


class TestHMM(unittest.TestCase):
    def test_viterbi(self):
        hmm = HMM(["A", "B"], add_k=0.1)
        hmm.train([[(1, "A"), (2, "B")], [(1, "A"), (2, "B")], [(2, "B"), (1, "A")]])
        path = hmm.viterbi([1, 2])
        self.assertEqual(len(path), 2)
        self.assertIn(path[0], ("A", "B"))


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_insert_query(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        store.insert_many([{"v": i} for i in range(25)])
        self.assertEqual(store.stats()["total"], 25)
        self.assertEqual(store.stats()["shard_count"], 3)
        self.assertEqual(len(store.query(where=[("v", "gt", 20)])), 4)
        self.assertEqual(len(store.query(where=[("v", "in", [1, 2, 3])])), 3)

    def test_delete_compact(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": i} for i in range(15)])
        store.delete(ids[0])
        stats = store.compact()
        self.assertEqual(stats["records"], 14)

    def test_concurrent_insert(self):
        store = ShardedStore(self.tmp, "t", shard_size=20)
        errors = []

        def worker(offset):
            try:
                store.insert_many([{"v": offset * 1000 + i} for i in range(30)])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(store.stats()["total"], 180)

    def test_registry_tasks(self):
        reg = StoreRegistry(self.tmp)
        reg.task("a").insert({"x": 1})
        reg.task("b").insert({"x": 2})
        # 造一个非存储目录，不应被识别为任务
        os.makedirs(os.path.join(self.tmp, "models"))
        self.assertEqual(reg.tasks(), ["a", "b"])

    def test_update_and_live_records(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        rid = store.insert({"name": "a", "tags": []})
        updated = store.update(rid, {"tags": ["x", "y"], "name": "b"})
        self.assertEqual(updated["tags"], ["x", "y"])
        self.assertEqual(updated["name"], "b")
        store.delete(rid)
        self.assertIsNone(store.get(rid))
        self.assertEqual(store.all(), [])


class TestCorpusService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ShardedStore(self.tmp, "corpus", shard_size=3)
        base = 1_700_000_000
        records = [
            {"name": "新闻甲", "text": "机器学习推动人工智能发展", "tags": ["新闻", "AI"],
             "created_at": base + 10},
            {"name": "评论乙", "text": "产品很好用，服务优秀", "tags": ["评论"],
             "created_at": base + 20},
            {"name": "论文丙", "text": "深度学习是人工智能的重要分支", "tags": ["论文", "AI"],
             "created_at": base + 30},
            {"name": "短消息", "text": "你好", "tags": ["消息"],
             "created_at": base + 40},
        ]
        self.ids = self.store.insert_many(records)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _params(self, **kwargs):
        args = {
            "tags": [], "tag_mode": "any", "keyword": "",
            "length_min": None, "length_max": None,
            "created_from": None, "created_to": None,
            "sort": "created_at", "order": "desc",
            "page_size": 2, "cursor": None, "direction": "next",
        }
        args.update(kwargs)
        return args

    def test_combined_filters(self):
        params = self._params(keyword="人工智能", tags=["AI"], tag_mode="all")
        result = query_corpus(self.store, params)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["records"][0]["id"], self.ids[2])

    def test_stable_cursor_pagination_no_duplicate_or_gap(self):
        expected_asc = self.ids
        expected_desc = list(reversed(self.ids))
        for order, expected in (("asc", expected_asc), ("desc", expected_desc)):
            params = self._params(sort="created_at", order=order, page_size=2)
            first = query_corpus(self.store, params)
            self.assertEqual([r["id"] for r in first["records"]], expected[:2])
            params["cursor"] = first["next_cursor"]
            second = query_corpus(self.store, params)
            self.assertEqual([r["id"] for r in second["records"]], expected[2:])
            params["cursor"] = second["prev_cursor"]
            params["direction"] = "prev"
            back = query_corpus(self.store, params)
            self.assertEqual([r["id"] for r in back["records"]], expected[:2])

            params["direction"] = "next"
            params["cursor"] = first["next_cursor"]
            seen = [r["id"] for r in first["records"]]
            page = first
            while page["next_cursor"]:
                params["cursor"] = page["next_cursor"]
                page = query_corpus(self.store, params)
                seen.extend(r["id"] for r in page["records"])
            self.assertEqual(sorted(seen), sorted(self.ids))

    def test_saved_filters_include_new_document(self):
        params = self._params(keyword="人工智能", tags=["AI"], tag_mode="all")
        before = query_corpus(self.store, params)
        self.assertEqual(before["total"], 2)
        self.store.insert({
            "name": "新文档", "text": "人工智能持续发展", "tags": ["AI"],
            "created_at": 1_700_000_100,
        })
        after = query_corpus(self.store, params)
        self.assertEqual(after["total"], 3)

    def test_delete_does_not_change_relative_order(self):
        params = self._params(sort="created_at", order="asc", page_size=10)
        before = [r["id"] for r in query_corpus(self.store, params)["records"]]
        self.store.delete(self.ids[1])
        after = [r["id"] for r in query_corpus(self.store, params)["records"]]
        expected = [i for i in before if i != self.ids[1]]
        self.assertEqual(after, expected)

    def test_export_uses_same_global_order_as_page(self):
        params = self._params(sort="created_at", order="desc", page_size=2)
        page = query_corpus(self.store, params)
        exported = query_corpus(self.store, params, full=True, page=False)
        self.assertEqual([r["id"] for r in exported["records"][:2]],
                         [r["id"] for r in page["records"]])
        self.assertEqual(exported["total"], 4)


class TestCorpusAPI(unittest.TestCase):
    def setUp(self):
        if create_app is None:
            self.skipTest("Flask 未安装")
        self.tmp = tempfile.mkdtemp()
        self.app = create_app(self.tmp)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_batch_upload_partial_failure_and_tags(self):
        payload = (
            '{"name":"good","text":"第一篇文档","tags":["新闻"]}\n'
            '{"broken":\n'
            '{"name":"empty","text":""}\n'
        ).encode("utf-8")
        resp = self.client.post(
            "/api/corpus/upload",
            data={"file": (io.BytesIO(payload), "docs.jsonl"), "tags": "AI"},
            content_type="multipart/form-data")
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["inserted_count"], 1)
        self.assertEqual(data["failed_count"], 2)
        self.assertEqual(data["inserted"][0]["tags"], ["AI", "新闻"])

    def test_filter_export_matches_page(self):
        for i in range(5):
            resp = self.client.post("/api/corpus", json={
                "name": f"新闻{i}", "text": f"关键词内容 {i}", "tags": ["新闻"]
            })
            self.assertEqual(resp.status_code, 200)
        resp = self.client.get("/api/corpus?tags=" + "%E6%96%B0%E9%97%BB"
                               + "&q=" + "%E5%85%B3%E9%94%AE%E8%AF%8D"
                               + "&page_size=2")
        page = resp.get_json()
        self.assertEqual(page["total"], 5)
        self.assertEqual(len(page["records"]), 2)
        export = self.client.get(
            "/api/corpus/export?tags=" + "%E6%96%B0%E9%97%BB"
            + "&q=" + "%E5%85%B3%E9%94%AE%E8%AF%8D"
            + "&scope=all&format=jsonl")
        lines = [line for line in export.data.decode("utf-8").splitlines() if line]
        self.assertEqual(len(lines), 5)
        ids = [json.loads(line)["id"] for line in lines]
        self.assertIn(page["records"][0]["id"], ids)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = PipelineEngine().register_builtin()

    def test_run_chain(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment"}, {"name": "pos"}, {"name": "sentiment"}]}
        out = self.engine.build(cfg).run({"text": "这个产品非常好用"})
        self.assertIn("words", out)
        self.assertIn("pos", out)
        self.assertIn("sentiment", out)

    def test_batch(self):
        cfg = {"name": "p", "stages": [{"name": "segment"}, {"name": "keywords"}]}
        results = self.engine.run_batch(
            cfg, ["今天天气很好", "这个产品非常好用"], max_workers=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["ok"] for r in results))

    def test_cycle_detected(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment", "deps": ["pos"]},
            {"name": "pos", "deps": ["segment"]},
        ]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)

    def test_missing_stage(self):
        cfg = {"name": "p", "stages": [{"name": "not_exist"}]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)


if __name__ == "__main__":
    unittest.main()
