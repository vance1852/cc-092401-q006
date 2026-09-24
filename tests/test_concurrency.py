"""并发幂等语义的确定性回归。

两个采集网关在网络恢复后可能用多个独立连接同时重放同一观测分片。
这些测试用同步屏障保证两个连接在同一时刻发起写入，依赖 SQLite 的
BEGIN IMMEDIATE 写锁串行化竞争事务，验证：

* 同作用域、同幂等键、同内容摘要：双方都拿到首个已提交响应；
* 同键不同内容：后提交方报告冲突，且不写入任何业务数据；
* 不同键但来源行重叠：仍然报告来源行重复，不会被误判为安全重放。
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from robot_trials.clock import FrozenClock
from robot_trials.errors import Conflict
from robot_trials.jsonio import load_json
from robot_trials.service import TrialService
from robot_trials.storage import connect

ROOT = Path(__file__).resolve().parents[1]


class ConcurrentImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "concurrent.sqlite3"
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        setup = TrialService(connect(self.db_path), self.clock)
        for user_id, role in (
            ("operator", "operator"),
            ("stat", "statistician"),
        ):
            setup.create_user(user_id, user_id, role)
        self.protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        self.rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        setup.register_robot("operator", "robot-a", "A 型", "厂商")
        setup.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        setup.publish_protocol("stat", self.protocol)
        setup.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        setup.start_batch("operator", "batch-a", 1)
        setup.connection.close()
        # 构造服务时会执行幂等的 schema 初始化；串行构造以避免 DDL 与写事务竞争，
        # 让屏障之后只发生业务事务的并发。
        self.init_lock = threading.Lock()
        self.barrier = threading.Barrier(2)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _run_concurrent(self, key_a: str, rows_a, key_b: str, rows_b) -> tuple[dict, dict]:
        """两个独立连接在同一屏障释放后并发导入，返回各自的 (status, payload)。"""

        results: dict[str, object] = {}

        def worker(name: str, key: str, rows) -> None:
            with self.init_lock:
                service = TrialService(
                    connect(self.db_path),
                    FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)),
                )
            self.barrier.wait()
            try:
                results[name] = ("ok", service.import_observations("operator", "batch-a", key, rows))
            except Conflict as exc:
                results[name] = ("conflict", str(exc))
            finally:
                service.connection.close()

        thread_a = threading.Thread(target=worker, args=("a", key_a, rows_a))
        thread_b = threading.Thread(target=worker, args=("b", key_b, rows_b))
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=15)
        thread_b.join(timeout=15)
        self.assertFalse(thread_a.is_alive() or thread_b.is_alive(), "并发导入发生死锁或忙等等待")
        return results["a"], results["b"]  # type: ignore[return-value]

    def _counts(self) -> tuple[int, int, int]:
        connection = sqlite3.connect(self.db_path)
        try:
            observations = connection.execute("SELECT count(*) FROM observations").fetchone()[0]
            keys = connection.execute(
                "SELECT count(*) FROM idempotency_keys WHERE scope=?",
                ("observations:batch-a",),
            ).fetchone()[0]
            events = connection.execute(
                "SELECT count(*) FROM audit_events "
                "WHERE entity_type='batch' AND entity_id='batch-a' AND event_type='observations.imported'"
            ).fetchone()[0]
        finally:
            connection.close()
        return observations, keys, events

    def test_same_key_same_content_both_commit_first_response(self) -> None:
        result_a, result_b = self._run_concurrent("gateway-key", self.rows, "gateway-key", self.rows)
        self.assertEqual(result_a[0], "ok", result_a)
        self.assertEqual(result_b[0], "ok", result_b)
        # 双方必须看到完全一致的首个已提交响应（含相同内容摘要与插入数）。
        self.assertEqual(result_a[1], result_b[1])
        self.assertEqual(result_a[1]["inserted"], 6)
        self.assertEqual(result_a[1]["request_sha256"], result_b[1]["request_sha256"])
        # 数据库只保留一份业务结果：一份观测、一个幂等记录、一个导入事件。
        self.assertEqual(self._counts(), (6, 1, 1))
        # 来源行唯一约束落点上也确实只有一份数据。
        connection = sqlite3.connect(self.db_path)
        try:
            identities = connection.execute(
                "SELECT source_batch, source_row FROM observations"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(len(identities), len(set(identities)))

    def test_same_key_different_content_reports_conflict_and_writes_nothing(self) -> None:
        changed = [dict(item) for item in self.rows]
        changed[0] = dict(changed[0])
        changed[0]["metrics"] = dict(changed[0]["metrics"])
        changed[0]["metrics"]["completion_seconds"] = "99"
        result_a, result_b = self._run_concurrent("gateway-key", self.rows, "gateway-key", changed)
        statuses = sorted((result_a[0], result_b[0]))
        # 一个成功提交，另一个必须报告“同键不同内容”冲突。
        self.assertEqual(statuses, ["conflict", "ok"])
        conflict_message = result_a[1] if result_a[0] == "conflict" else result_b[1]
        self.assertIn("不同请求内容", conflict_message)
        # 输家不得产生任何观测、幂等记录或导入事件。
        self.assertEqual(self._counts(), (6, 1, 1))

    def test_different_keys_overlapping_source_rows_still_conflict(self) -> None:
        # A 导入 001/002，B 用另一个幂等键导入 002/003：
        # 这不是安全重放，输家必须报来源行重复且自己的 003 不得残留（无部分事务）。
        result_a, result_b = self._run_concurrent(
            "key-a", self.rows[:2], "key-b", self.rows[1:3]
        )
        statuses = sorted((result_a[0], result_b[0]))
        self.assertEqual(statuses, ["conflict", "ok"])
        conflict_message = result_a[1] if result_a[0] == "conflict" else result_b[1]
        self.assertIn("来源行重复", conflict_message)
        # 只保留赢家的两条观测；输家那条不重叠的 003 必须随事务回滚消失。
        observations, keys, events = self._counts()
        self.assertEqual(observations, 2)
        self.assertEqual(keys, 1)
        self.assertEqual(events, 1)
        connection = sqlite3.connect(self.db_path)
        try:
            source_rows = {
                row[0]
                for row in connection.execute("SELECT source_row FROM observations").fetchall()
            }
        finally:
            connection.close()
        # 赢家的数据是 {001,002} 或 {002,003} 之一；输家独占的来源行必须随回滚消失。
        self.assertIn(source_rows, [{"001", "002"}, {"002", "003"}])
        self.assertNotIn({"001", "002", "003"}, source_rows)


if __name__ == "__main__":
    unittest.main()
