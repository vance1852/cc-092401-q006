"""多个独立连接并发导入观测分片的确定性回归测试。"""

from __future__ import annotations

import json
import sqlite3
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from robot_trials.clock import FrozenClock
from robot_trials.errors import Conflict
from robot_trials.jsonio import content_digest, load_json
from robot_trials.service import TrialService
from robot_trials.storage import connect


ROOT = Path(__file__).resolve().parents[1]


class _SynchronizedService(TrialService):
    """在首次幂等预检后等待屏障，迫使两个网关都完成预检再竞争写锁。

    这样重放竞争是确定的：双方都看到幂等键不存在，随后一方拿到
    SQLite 写锁先提交，另一方在 BEGIN IMMEDIATE 上等待，直到对方
    提交后才继续，从而稳定复现网络恢复后同时重放的交错时序。
    """

    def __init__(self, connection: sqlite3.Connection, clock, barrier: threading.Barrier) -> None:
        super().__init__(connection, clock)
        self._barrier = barrier
        self._pre_check_done = False

    def _idempotent_response(self, scope, key, request_digest):
        result = super()._idempotent_response(scope, key, request_digest)
        if not self._pre_check_done:
            self._pre_check_done = True
            self._barrier.wait(timeout=10)
        return result


class ConcurrentImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "trials.sqlite3"
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        service = TrialService(connect(self.db_path), self.clock)
        service.create_user("operator", "operator", "operator")
        service.create_user("stat", "stat", "statistician")
        service.register_robot("operator", "robot-a", "A 型", "厂商")
        service.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        service.publish_protocol("stat", load_json(ROOT / "fixtures" / "demo_protocol.json"))
        service.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        service.start_batch("operator", "batch-a", 1)
        service.connection.close()
        self.rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _run_pair(self, first: tuple[str, list], second: tuple[str, list]) -> dict[str, tuple[str, object]]:
        """两个网关各自使用独立连接，经同步屏障同时导入同一批次。"""

        barrier = threading.Barrier(2)
        outcomes: dict[str, tuple[str, object]] = {}

        def worker(name: str, key: str, rows: list) -> None:
            service = _SynchronizedService(connect(self.db_path), self.clock, barrier)
            try:
                outcomes[name] = ("ok", service.import_observations("operator", "batch-a", key, rows))
            except Conflict as exc:
                outcomes[name] = ("conflict", exc)
            except BaseException as exc:  # 让主线程报告意外错误
                outcomes[name] = ("error", exc)
            finally:
                service.connection.close()

        threads = [
            threading.Thread(target=worker, args=("first", *first)),
            threading.Thread(target=worker, args=("second", *second)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        for thread in threads:
            self.assertFalse(thread.is_alive(), "并发导入线程未在限期内结束")
        self.assertEqual(set(outcomes), {"first", "second"})
        for name, (kind, payload) in outcomes.items():
            if kind == "error":
                self.fail(f"网关 {name} 出现意外错误: {payload!r}")
        return outcomes

    def _database_state(self) -> dict[str, object]:
        connection = connect(self.db_path)
        try:
            observations = connection.execute("SELECT count(*) FROM observations").fetchone()[0]
            keys = connection.execute("SELECT count(*) FROM idempotency_keys").fetchone()[0]
            events = connection.execute(
                "SELECT count(*) FROM audit_events WHERE event_type='observations.imported'"
            ).fetchone()[0]
            stored = connection.execute(
                "SELECT request_sha256,response_json FROM idempotency_keys "
                "WHERE scope='observations:batch-a' AND key='key-1'"
            ).fetchone()
            return {
                "observations": observations,
                "idempotency_keys": keys,
                "import_events": events,
                "stored": None if stored is None else (stored[0], json.loads(stored[1])),
            }
        finally:
            connection.close()

    def test_concurrent_identical_replay_returns_first_committed_response(self) -> None:
        outcomes = self._run_pair(("key-1", self.rows), ("key-1", self.rows))
        first = outcomes["first"]
        second = outcomes["second"]
        self.assertEqual(first[0], "ok", f"首个网关应成功: {first[1]!r}")
        self.assertEqual(second[0], "ok", f"重放网关应拿到首个已提交响应而不是冲突: {second[1]!r}")
        self.assertEqual(first[1], second[1])
        self.assertEqual(
            first[1],
            {"batch_id": "batch-a", "inserted": len(self.rows), "request_sha256": content_digest(self.rows)},
        )
        state = self._database_state()
        self.assertEqual(state["observations"], len(self.rows), "观测只能保留一份")
        self.assertEqual(state["idempotency_keys"], 1, "幂等键只能记录一次")
        self.assertEqual(state["import_events"], 1, "导入审计事件不能重复")
        self.assertEqual(state["stored"], (content_digest(self.rows), first[1]))

    def test_concurrent_same_rows_different_keys_still_conflict(self) -> None:
        outcomes = self._run_pair(("key-1", self.rows), ("key-2", self.rows))
        kinds = sorted(outcome[0] for outcome in outcomes.values())
        self.assertEqual(kinds, ["conflict", "ok"], "真实来源行冲突必须报告冲突")
        conflict = next(payload for kind, payload in outcomes.values() if kind == "conflict")
        self.assertIn("来源行重复", str(conflict))
        state = self._database_state()
        self.assertEqual(state["observations"], len(self.rows))
        self.assertEqual(state["idempotency_keys"], 1)
        self.assertEqual(state["import_events"], 1)

    def test_concurrent_same_key_different_content_conflicts(self) -> None:
        changed = [dict(item) for item in self.rows]
        changed[0] = dict(changed[0])
        changed[0]["metrics"] = dict(changed[0]["metrics"])
        changed[0]["metrics"]["completion_seconds"] = "99"
        outcomes = self._run_pair(("key-1", self.rows), ("key-1", changed))
        kinds = sorted(outcome[0] for outcome in outcomes.values())
        self.assertEqual(kinds, ["conflict", "ok"], "同键不同内容必须报告冲突")
        conflict = next(payload for kind, payload in outcomes.values() if kind == "conflict")
        self.assertIn("不同请求内容", str(conflict))
        ok = next(payload for kind, payload in outcomes.values() if kind == "ok")
        state = self._database_state()
        self.assertEqual(state["observations"], len(self.rows), "只能保留胜出版本的观测")
        self.assertEqual(state["idempotency_keys"], 1)
        self.assertEqual(state["import_events"], 1)
        self.assertEqual(state["stored"], (ok["request_sha256"], ok))

    def test_replay_from_another_connection_after_commit(self) -> None:
        first_service = TrialService(connect(self.db_path), self.clock)
        first = first_service.import_observations("operator", "batch-a", "key-1", self.rows)
        first_service.connection.close()
        second_service = TrialService(connect(self.db_path), self.clock)
        second = second_service.import_observations("operator", "batch-a", "key-1", self.rows)
        second_service.connection.close()
        self.assertEqual(first, second)
        state = self._database_state()
        self.assertEqual(state["observations"], len(self.rows))
        self.assertEqual(state["idempotency_keys"], 1)
        self.assertEqual(state["import_events"], 1)


if __name__ == "__main__":
    unittest.main()
