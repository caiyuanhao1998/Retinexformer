"""Persistent aggregate success counters for the hosted Gradio demos."""

import copy
import functools
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

LOG = logging.getLogger(__name__)
TIMEZONE = "America/Los_Angeles"
CHANNELS = {"retinexformer", "diffusiongs-object", "diffusiongs-scene"}


class HubStore:
    """Use immutable reads and optimistic commits to avoid lost increments."""

    def __init__(self, repo_id, token):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)

    def read(self):
        info = self.api.dataset_info(self.repo_id, timeout=8)
        if not info.private:
            raise ValueError("Success counters require a private dataset")
        url = f"https://huggingface.co/datasets/{self.repo_id}/raw/{info.sha}/counts.json"
        request = Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urlopen(request, timeout=8) as response:
            data = json.load(response)
        if data.get("schema_version") != 1 or data.get("timezone") != TIMEZONE:
            raise ValueError("Unexpected counter schema or timezone")
        return data, info.sha

    def commit(self, data, parent):
        from huggingface_hub import CommitOperationAdd
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            parent_commit=parent,
            commit_message="Update aggregate Space success counters",
            operations=[CommitOperationAdd(
                path_in_repo="counts.json",
                path_or_fileobj=(json.dumps(data, indent=2) + "\n").encode(),
            )],
        )


class SuccessStats:
    def __init__(self, store, *, attempts=5, cache_seconds=30):
        self.store = store
        self.attempts = attempts
        self.cache_seconds = cache_seconds
        self.lock = threading.RLock()
        self.cached = None
        self.cache_time = 0

    @classmethod
    def from_env(cls):
        token = os.environ.get("SPACE_STATS_TOKEN")
        repo = os.environ.get("SPACE_STATS_REPO", "CaiYuanhao/space-success-stats")
        return cls(HubStore(repo, token) if token else None)

    def _update(self, mutate):
        if self.store is None:
            raise RuntimeError("SPACE_STATS_TOKEN has not been configured")
        with self.lock:
            for attempt in range(self.attempts):
                try:
                    data, revision = self.store.read()
                    changed = mutate(data)
                    if changed:
                        self.store.commit(data, revision)
                    self.cached = copy.deepcopy(data)
                    self.cache_time = time.monotonic()
                    return
                except Exception as error:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if status is None:
                        status = getattr(error, "code", None)
                    if status in (400, 401, 403, 404) or attempt + 1 == self.attempts:
                        raise
                    # A timeout can occur after the commit was accepted. Re-read
                    # and check the same event ID before attempting an increment.
                    time.sleep(0.1 * (2 ** attempt))

    def start(self, *channels, now=None):
        now = now or datetime.now(timezone.utc)
        if not set(channels) <= CHANNELS:
            raise ValueError("Unknown counter channel")

        def mutate(data):
            changed = False
            for channel in channels:
                if channel not in data["started_at"]:
                    data["started_at"][channel] = now.isoformat()
                    changed = True
            return changed

        try:
            self._update(mutate)
            return True
        except Exception as error:
            LOG.warning("Success statistics unavailable during initialization (%s)", type(error).__name__)
            return False

    def record(self, channel, *, now=None, event_id=None):
        if channel not in CHANNELS:
            raise ValueError("Unknown counter channel")
        now = now or datetime.now(timezone.utc)
        day = now.astimezone(ZoneInfo(TIMEZONE)).date().isoformat()
        event_id = event_id or str(uuid.uuid4())

        def mutate(data):
            if event_id in data["recent_event_ids"]:
                return False
            data["started_at"].setdefault(channel, now.isoformat())
            counts = data["days"].setdefault(day, {})
            counts[channel] = counts.get(channel, 0) + 1
            data["recent_event_ids"] = (data["recent_event_ids"] + [event_id])[-1024:]
            return True

        try:
            self._update(mutate)
            return True
        except Exception as error:
            # A metrics outage must not discard an already generated result.
            # Keep the anonymous event in logs for recovery, without user data.
            LOG.warning("SUCCESS_COUNT_NOT_SAVED channel=%s day=%s event_id=%s error=%s",
                        channel, day, event_id, type(error).__name__)
            self.cached = None
            return False

    def successful(self, channel):
        """Wrap the CPU caller of a GPU function, never the forked GPU body."""
        def decorate(function):
            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                result = function(*args, **kwargs)
                self.record(channel)
                return result
            return wrapped
        return decorate

    def snapshot(self):
        if self.store is None:
            raise RuntimeError("Success statistics have not been configured")
        with self.lock:
            if self.cached is None or time.monotonic() - self.cache_time > self.cache_seconds:
                self.cached, _ = self.store.read()
                self.cache_time = time.monotonic()
            return copy.deepcopy(self.cached)

    def view(self, channel, *, now=None):
        try:
            data = self.snapshot()
        except Exception:
            return "Success counts are temporarily unavailable.", []
        return summarize(data, channel, now=now)


def summarize(data, channel, *, now=None):
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(TIMEZONE))
    month_start = local.date().replace(day=1)
    previous_end = month_start - timedelta(days=1)
    previous_start = previous_end.replace(day=1)
    label = previous_start.strftime("%Y-%m")
    start = data["started_at"].get(channel)
    if not start:
        return f"**Last month ({label}):** No historical data. Tracking has not started.", []
    started = datetime.fromisoformat(start).astimezone(ZoneInfo(TIMEZONE))
    if started.date() > previous_end:
        total_text = "No historical data"
    else:
        total = sum(values.get(channel, 0) for day, values in data["days"].items()
                    if previous_start.isoformat() <= day <= previous_end.isoformat())
        total_text = f"{total:,} successful runs"
        beginning = datetime.combine(previous_start, datetime.min.time(), tzinfo=ZoneInfo(TIMEZONE))
        if started > beginning:
            total_text += " (partial month)"
    rows = []
    day = local.date()
    earliest = max(previous_start, started.date())
    while day >= earliest:
        rows.append([day.isoformat(), data["days"].get(day.isoformat(), {}).get(channel, 0)])
        day -= timedelta(days=1)
    summary = (f"**Last month ({label}):** {total_text}.\n\n"
               f"Daily counts use {TIMEZONE}. Tracking started {started:%Y-%m-%d %H:%M %Z}. "
               "Earlier activity is not included. Only completed enhancement or generation runs count.")
    return summary, rows


def add_usage_panel(demo, stats, channel):
    import gradio as gr
    with demo:
        with gr.Accordion("Successful runs", open=False):
            summary = gr.Markdown("Loading success counts…")
            daily = gr.Dataframe(headers=[f"Date ({TIMEZONE})", "Successful runs"],
                                 datatype=["str", "number"], value=[], interactive=False,
                                 label="Daily successful runs", row_count=(0, "dynamic"))
            refresh = gr.Button("Refresh counts", size="sm")

        def load_counts():
            return stats.view(channel)

        demo.load(load_counts, outputs=[summary, daily], queue=False, api_name=False)
        refresh.click(load_counts, outputs=[summary, daily], queue=False, api_name=False)
