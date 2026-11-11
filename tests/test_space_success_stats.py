import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demo/huggingface'))

import copy
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import patch

from space_success_stats import SuccessStats, summarize, success_badges


def state():
    return {"schema_version": 1, "timezone": "America/Los_Angeles", "started_at": {},
            "days": {}, "recent_event_ids": []}


class MemoryStore:
    def __init__(self):
        self.data = state()
        self.revision = 0
        self.lock = threading.Lock()
        self.timeout_after_commit = False

    def read(self):
        with self.lock:
            return copy.deepcopy(self.data), self.revision

    def commit(self, data, parent):
        with self.lock:
            if parent != self.revision:
                raise RuntimeError("Concurrent update")
            self.data = copy.deepcopy(data)
            self.revision += 1
            if self.timeout_after_commit:
                self.timeout_after_commit = False
                raise TimeoutError("Response lost after successful commit")


class CountersTest(unittest.TestCase):
    def test_configurable_secret_name(self):
        store = MemoryStore()
        for settings in (
            {"SPACE_STATS_TOKEN": "test-only-token"},
            {"SPACE_STATS_TOKEN_NAME": "custom_stats_secret", "custom_stats_secret": "test-only-token"},
        ):
            with patch.dict("os.environ", settings, clear=True):
                with patch("space_success_stats.HubStore", return_value=store) as constructor:
                    self.assertIs(SuccessStats.from_env().store, store)
                    constructor.assert_called_once_with("CaiYuanhao/space-success-stats", "test-only-token")

    def test_success_only_and_return_value_preserved(self):
        store = MemoryStore()
        stats = SuccessStats(store)

        @stats.successful("retinexformer")
        def process(valid):
            if not valid:
                raise ValueError("Inference failed")
            return ("image", "details")

        with self.assertRaises(ValueError):
            process(False)
        self.assertEqual(store.data["days"], {})
        self.assertEqual(process(True), ("image", "details"))
        self.assertEqual(sum(d["retinexformer"] for d in store.data["days"].values()), 1)

    def test_uncertain_commit_is_not_counted_twice(self):
        store = MemoryStore()
        store.timeout_after_commit = True
        stats = SuccessStats(store)
        with patch("space_success_stats.time.sleep"):
            self.assertTrue(stats.record("diffusiongs-scene", event_id="request-1"))
            self.assertTrue(stats.record("diffusiongs-scene", event_id="request-1"))
        self.assertEqual(sum(d["diffusiongs-scene"] for d in store.data["days"].values()), 1)

    def test_multiple_writers_preserve_all_successes(self):
        store = MemoryStore()
        writers = [SuccessStats(store, attempts=100) for _ in range(8)]
        with patch("space_success_stats.time.sleep"):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda i: writers[i % 8].record(
                    "diffusiongs-object", event_id=f"request-{i}"), range(40)))
        self.assertTrue(all(results))
        self.assertEqual(sum(d["diffusiongs-object"] for d in store.data["days"].values()), 40)
        self.assertNotIn("diffusiongs-scene", next(iter(store.data["days"].values())))

    def test_la_midnight_uses_local_date(self):
        store = MemoryStore()
        stats = SuccessStats(store)
        stats.record("retinexformer", now=datetime(2026, 11, 10, 7, 59, tzinfo=timezone.utc))
        stats.record("retinexformer", now=datetime(2026, 11, 10, 8, 1, tzinfo=timezone.utc))
        self.assertEqual(store.data["days"]["2026-11-09"]["retinexformer"], 1)
        self.assertEqual(store.data["days"]["2026-11-10"]["retinexformer"], 1)

    def test_new_tracking_includes_this_month_without_waiting(self):
        data = state()
        data["started_at"]["retinexformer"] = "2026-11-10T12:00:00+00:00"
        data["days"] = {"2026-11-10": {"retinexformer": 6}, "2026-11-11": {"retinexformer": 18}}
        when = datetime(2026, 11, 11, 20, tzinfo=timezone.utc)
        summary, rows = summarize(data, "retinexformer", now=when)
        self.assertIn("24 successful runs", summary)
        self.assertEqual(rows, [["2026-11-11", 18], ["2026-11-10", 6]])
        badges = success_badges(data, "retinexformer", now=when)
        self.assertIn("Last 30 days", badges)
        self.assertIn(">24</strong>", badges)
        self.assertIn(">18</strong>", badges)

    def test_rolling_window_includes_exactly_30_local_dates(self):
        data = state()
        data["started_at"]["retinexformer"] = "2026-09-01T00:00:00+00:00"
        data["days"] = {"2026-10-12": {"retinexformer": 100},
                        "2026-10-13": {"retinexformer": 2, "diffusiongs-object": 900},
                        "2026-11-10": {"retinexformer": 3},
                        "2026-11-11": {"retinexformer": 4},
                        "2026-11-12": {"retinexformer": 1000}}
        summary, rows = summarize(data, "retinexformer", now=datetime(2026, 11, 11, 20, tzinfo=timezone.utc))
        self.assertIn("9 successful runs", summary)
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0], ["2026-11-11", 4])
        self.assertEqual(rows[-1], ["2026-10-13", 2])

    def test_rolling_window_crosses_year_boundary(self):
        data = state()
        data["started_at"]["retinexformer"] = "2026-12-01T12:00:00+00:00"
        data["days"] = {"2026-12-03": {"retinexformer": 100},
                        "2026-12-04": {"retinexformer": 2},
                        "2026-12-20": {"retinexformer": 7},
                        "2027-01-02": {"retinexformer": 3}}
        summary, rows = summarize(data, "retinexformer", now=datetime(2027, 1, 2, 20, tzinfo=timezone.utc))
        self.assertIn("12 successful runs", summary)
        self.assertEqual(rows[-1], ["2026-12-04", 2])

    def test_measured_zero_is_distinct_from_untracked_history(self):
        data = state()
        when = datetime(2026, 11, 11, 20, tzinfo=timezone.utc)
        badges = success_badges(data, "retinexformer", now=when)
        self.assertIn(">—</strong>", badges)
        data["started_at"]["retinexformer"] = "2026-11-11T12:00:00+00:00"
        badges = success_badges(data, "retinexformer", now=when)
        self.assertNotIn(">—</strong>", badges)
        self.assertEqual(badges.count(">0</strong>"), 2)

    def test_new_process_recovers_persisted_counts(self):
        store = MemoryStore()
        SuccessStats(store).record("retinexformer")
        self.assertEqual(SuccessStats(store).snapshot(), store.data)

    def test_unconfigured_statistics_do_not_break_inference(self):
        stats = SuccessStats(None, attempts=1)
        with self.assertLogs("space_success_stats", level="WARNING"):
            self.assertEqual(stats.successful("retinexformer")(lambda: "result")(), "result")
        self.assertEqual(stats.view("retinexformer"), ("Success counts are temporarily unavailable.", []))


if __name__ == "__main__":
    unittest.main()
