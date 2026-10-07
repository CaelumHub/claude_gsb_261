"""语料库管理（批量入库 / 标签 / 组合筛选 / 导出）测试。

运行：``python -m unittest discover -s tests -v``
覆盖：
- 存储层：update 原位更新保序、稳定排序跨分片分页不重不漏、墓碑排除
- API：批量导入容错、多文件上传容错、标签 CRUD、组合筛选、
  分页连续性、导出与页面一致、筛选条件持久化、删除隔离性
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from storage import ShardedStore


class TestStoreUpdateAndOrder(unittest.TestCase):
    """存储层：原位更新、稳定排序、跨分片分页。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_update_preserves_position(self):
        store = ShardedStore(self.tmp, "t", shard_size=5)
        ids = store.insert_many([{"v": i, "tags": []} for i in range(9)])
        before = [r["id"] for r in store.all()]
        updated = store.update(ids[4], {"tags": ["x"]})
        self.assertEqual(updated["tags"], ["x"])
        after = [r["id"] for r in store.all()]
        self.assertEqual(before, after)  # 位置不变
        # id / _seq 不可被 patch 覆盖
        again = store.update(ids[4], {"id": "hack", "_seq": 999})
        self.assertEqual(again["id"], ids[4])
        self.assertNotEqual(again["_seq"], 999)

    def test_update_missing_returns_none(self):
        store = ShardedStore(self.tmp, "t", shard_size=5)
        self.assertIsNone(store.update("nope", {"a": 1}))
        rid = store.insert({"a": 1})
        store.delete(rid)
        self.assertIsNone(store.update(rid, {"a": 2}))  # 墓碑不可更新

    def test_stable_order_pagination_no_dup_no_miss(self):
        """字段值全部相同时，分页拼接必须与全量排序一致（不重不漏）。"""
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": 1} for _ in range(25)])  # 排序键全相同
        full = [r["id"] for r in store.query(order_by="v", order="desc")]
        self.assertEqual(len(full), 25)
        paged = []
        for page in range(3):  # 10 + 10 + 5
            chunk = store.query(order_by="v", order="desc",
                                limit=10, offset=page * 10)
            paged.extend(r["id"] for r in chunk)
        self.assertEqual(paged, full)
        self.assertEqual(sorted(paged), sorted(ids))

    def test_query_excludes_tombstones(self):
        store = ShardedStore(self.tmp, "t", shard_size=5)
        ids = store.insert_many([{"v": i} for i in range(7)])
        store.delete(ids[2])
        got = store.query()
        self.assertEqual(len(got), 6)
        self.assertNotIn(ids[2], [r["id"] for r in got])
        # 删除不影响其它记录相对顺序
        rest = [r["id"] for r in got]
        self.assertEqual(rest, [i for i in ids if i != ids[2]])


class TestCorpusAPI(unittest.TestCase):
    """API 层：批量导入 / 标签 / 筛选 / 分页 / 导出 / 筛选条件保存。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = create_app(self.tmp)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- 工具 ---------------------------------------------------------------
    def _batch(self, docs):
        return self.client.post("/api/corpus/batch",
                                json={"documents": docs})

    def _query(self, **params):
        resp = self.client.get("/api/corpus/query?" +
                               "&".join(f"{k}={v}" for k, v in params.items()))
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def _all_pages(self, page_size=10, **params):
        """逐页取完，返回拼接后的 id 列表。"""
        ids, page = [], 1
        while True:
            data = self._query(page=page, page_size=page_size, **params)
            ids.extend(it["id"] for it in data["items"])
            if page >= data["pages"]:
                return ids, data["total"]
            page += 1

    # -- 批量导入 -----------------------------------------------------------
    def test_batch_partial_failure(self):
        resp = self._batch([
            {"name": "ok1", "text": "第一篇", "tags": ["新闻"]},
            {"name": "bad1"},                                  # 缺 text
            {"name": "ok2", "text": "第二篇"},
            {"text": 123},                                     # text 类型错
            "not-a-dict",                                      # 整体非法
            {"name": "ok3", "text": "第三篇", "tags": "新闻, 财经"},
        ])
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["imported"], 3)
        self.assertEqual(data["failed"], 3)
        self.assertEqual([e["index"] for e in data["errors"]], [1, 3, 4])
        # 合法文档真实入库
        listed = self.client.get("/api/corpus").get_json()["corpora"]
        self.assertEqual(len(listed), 3)
        # 字符串形式的 tags 被正确拆分
        tags = {t["name"] for t in
                self.client.get("/api/corpus/tags").get_json()["tags"]}
        self.assertEqual(tags, {"新闻", "财经"})

    def test_batch_rejects_non_list(self):
        resp = self.client.post("/api/corpus/batch", json={"documents": {}})
        self.assertEqual(resp.status_code, 400)

    def test_upload_batch_partial_failure(self):
        good = (io.BytesIO("好文档".encode("utf-8")), "a.txt", "text/plain")
        bad = (io.BytesIO(b"\xff\xfe\x00\x8f\x8f"), "b.bin",
               "application/octet-stream")
        empty = (io.BytesIO(b"   "), "c.txt", "text/plain")
        resp = self.client.post(
            "/api/corpus/upload_batch",
            data={"files": [good, bad, empty]},
            content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["imported"], 1)
        self.assertEqual(data["failed"], 2)
        names = {e["name"] for e in data["errors"]}
        self.assertEqual(names, {"b.bin", "c.txt"})

    # -- 标签 ---------------------------------------------------------------
    def test_tags_update_and_isolation(self):
        ids = self._batch([
            {"name": "d1", "text": "一", "tags": ["甲"]},
            {"name": "d2", "text": "二", "tags": ["乙"]},
        ]).get_json()["ids"]
        resp = self.client.put(f"/api/corpus/{ids[0]}/tags",
                               json={"tags": ["甲", "丙", "丙", " "]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["tags"], ["甲", "丙"])  # 去重去空
        # 另一篇不受影响
        other = self.client.get(f"/api/corpus/{ids[1]}").get_json()
        self.assertEqual(other["tags"], ["乙"])
        # 不存在的 id
        resp = self.client.put("/api/corpus/corpus_999/tags", json={"tags": []})
        self.assertEqual(resp.status_code, 404)

    # -- 组合筛选 -----------------------------------------------------------
    def _seed(self):
        docs = []
        for i in range(30):
            docs.append({
                "name": f"文档{i:02d}",
                "text": ("人工智能" if i % 3 == 0 else "天气预报") * (i + 1),
                "tags": (["科技"] if i % 2 == 0 else []) + (["头条"] if i % 5 == 0 else []),
                "created_at": 1700000000 + i * 100,
            })
        return self._batch(docs).get_json()["ids"]

    def test_filter_by_tags_any_all(self):
        self._seed()
        any_hit = self._query(tags="科技,头条", tag_mode="any", page_size=100)
        # 偶数(科技) 或 5 的倍数(头条)
        expect_any = [i for i in range(30) if i % 2 == 0 or i % 5 == 0]
        self.assertEqual(any_hit["total"], len(expect_any))
        all_hit = self._query(tags="科技,头条", tag_mode="all", page_size=100)
        expect_all = [i for i in range(30) if i % 2 == 0 and i % 5 == 0]
        self.assertEqual(all_hit["total"], len(expect_all))

    def test_filter_by_keyword_length_time(self):
        self._seed()
        kw = self._query(keyword="人工智能", page_size=100)
        self.assertEqual(kw["total"], 10)  # i % 3 == 0
        # 长度：第 i 篇长度 = 4*(i+1) 或 4*(i+1)（两个词都是 4 字）
        ln = self._query(min_len=40, max_len=60, page_size=100)
        expect = [i for i in range(30) if 40 <= 4 * (i + 1) <= 60]
        self.assertEqual(ln["total"], len(expect))
        # 时间窗口
        tm = self._query(start=1700001000, end=1700001500, page_size=100)
        expect_t = [i for i in range(30)
                    if 1700001000 <= 1700000000 + i * 100 <= 1700001500]
        self.assertEqual(tm["total"], len(expect_t))
        # 组合：关键词 + 标签 + 长度
        combo = self._query(keyword="人工智能", tags="科技",
                            min_len=8, page_size=100)
        expect_c = [i for i in range(30)
                    if i % 3 == 0 and i % 2 == 0 and 4 * (i + 1) >= 8]
        self.assertEqual(combo["total"], len(expect_c))

    def test_sort_by_length_and_name(self):
        self._seed()
        asc = self._query(sort="length", order="asc", page_size=100)
        lens = [it["length"] for it in asc["items"]]
        self.assertEqual(lens, sorted(lens))
        by_name = self._query(sort="name", order="asc", page_size=100)
        names = [it["name"] for it in by_name["items"]]
        self.assertEqual(names, sorted(names))

    # -- 跨分片分页：连续、不重不漏 ----------------------------------------
    def test_pagination_across_shards(self):
        # 默认 shard_size=100，插入 105 篇保证跨分片；created_at 全相同制造并列
        docs = [{"name": f"d{i}", "text": f"正文{i}", "created_at": 1700000000}
                for i in range(105)]
        self._batch(docs)
        full = self._query(page=1, page_size=200, sort="created_at", order="desc")
        self.assertEqual(full["total"], 105)
        full_ids = [it["id"] for it in full["items"]]
        paged_ids, total = self._all_pages(page_size=10, sort="created_at",
                                           order="desc")
        self.assertEqual(total, 105)
        self.assertEqual(paged_ids, full_ids)          # 顺序一致
        self.assertEqual(len(set(paged_ids)), 105)     # 不重
        # 升序同样连续
        paged_asc, _ = self._all_pages(page_size=7, sort="created_at",
                                       order="asc")
        self.assertEqual(paged_asc, list(reversed(full_ids)))

    def test_delete_keeps_order_of_others(self):
        docs = [{"name": f"d{i}", "text": f"正文{i}", "tags": [f"t{i % 3}"]}
                for i in range(25)]
        ids = self._batch(docs).get_json()["ids"]
        before, _ = self._all_pages(page_size=100, sort="created_at", order="asc")
        # 删除中间一篇
        self.client.delete(f"/api/corpus/{ids[12]}")
        after, total = self._all_pages(page_size=6, sort="created_at", order="asc")
        self.assertEqual(total, 24)
        self.assertEqual(after, [i for i in before if i != ids[12]])
        # 其它文档标签不受影响
        tags = self.client.get("/api/corpus/tags").get_json()["tags"]
        self.assertEqual(sum(t["count"] for t in tags), 24)

    # -- 导出与页面一致 ------------------------------------------------------
    def test_export_matches_page(self):
        self._seed()
        params = {"keyword": "人工智能", "tags": "科技", "sort": "length",
                  "order": "asc"}
        page = self._query(page_size=100, **params)
        resp = self.client.post("/api/corpus/export",
                                json={"filter": params, "format": "json"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp.headers["Content-Disposition"])
        exported = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(exported["count"], page["total"])
        self.assertEqual([d["id"] for d in exported["documents"]],
                         [it["id"] for it in page["items"]])
        # 导出包含全文，页面只有预览
        self.assertTrue(all("text" in d for d in exported["documents"]))
        # TXT 导出
        resp = self.client.post("/api/corpus/export",
                                json={"filter": params, "format": "txt"})
        self.assertEqual(resp.status_code, 200)
        body = resp.data.decode("utf-8")
        self.assertIn("===== 文档00", body)
        self.assertIn("标签: 科技", body)

    # -- 筛选条件保存 --------------------------------------------------------
    def test_saved_filters_persist_and_stay_effective(self):
        self._seed()
        filt = {"tags": ["科技"], "keyword": "人工智能", "sort": "length",
                "order": "asc"}
        resp = self.client.post("/api/corpus/filters",
                                json={"name": "AI 科技", "filter": filt})
        self.assertEqual(resp.status_code, 200)
        fid = resp.get_json()["id"]

        # 模拟重启：同一数据目录重建 app，筛选条件仍在
        app2 = create_app(self.tmp)
        client2 = app2.test_client()
        filters = client2.get("/api/corpus/filters").get_json()["filters"]
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0]["filter"]["tags"], ["科技"])

        # 新入库一篇命中条件的文档 + 把一篇不命中的旧文档改标签后，条件依然生效
        before = client2.get("/api/corpus/query?tags=科技&keyword=人工智能"
                             "&page_size=100").get_json()["total"]
        client2.post("/api/corpus/batch", json={"documents": [
            {"name": "新AI", "text": "人工智能新突破", "tags": ["科技"]}]})
        victim = next(c for c in client2.get("/api/corpus").get_json()["corpora"]
                      if "科技" not in c["tags"])
        client2.put(f"/api/corpus/{victim['id']}/tags", json={"tags": ["生活"]})
        after = client2.get("/api/corpus/query?tags=科技&keyword=人工智能"
                            "&page_size=100").get_json()
        self.assertEqual(after["total"], before + 1)
        self.assertIn("新AI", [it["name"] for it in after["items"]])

        # 删除
        resp = client2.delete(f"/api/corpus/filters/{fid}")
        self.assertTrue(resp.get_json()["ok"])
        self.assertEqual(client2.get("/api/corpus/filters")
                         .get_json()["filters"], [])

    def test_saved_filter_validation(self):
        resp = self.client.post("/api/corpus/filters",
                                json={"name": "", "filter": {}})
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post("/api/corpus/filters",
                                json={"name": "x", "filter": "not-dict"})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
