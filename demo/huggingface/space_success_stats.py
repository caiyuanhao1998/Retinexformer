"""Persistent aggregate success counters for the hosted Gradio demos."""

import copy
import functools
import html
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
        token_name = os.environ.get("SPACE_STATS_TOKEN_NAME", "SPACE_STATS_TOKEN")
        token = os.environ.get(token_name)
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


def rolling_period(data, channel, *, now=None):
    """Today and the preceding 29 local dates, using the stored daily counts."""
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(TIMEZONE))
    first_day = local.date() - timedelta(days=29)
    start = data["started_at"].get(channel) if data else None
    if not start:
        return local, first_day, None, None
    started = datetime.fromisoformat(start).astimezone(ZoneInfo(TIMEZONE))
    total = sum(values.get(channel, 0) for day, values in data["days"].items()
                if first_day.isoformat() <= day <= local.date().isoformat())
    return local, first_day, started, total


def summarize(data, channel, *, now=None):
    local, first_day, started, total = rolling_period(data, channel, now=now)
    if started is None:
        return "**Last 30 days:** No historical data. Tracking has not started.", []
    rows = []
    day = local.date()
    earliest = max(first_day, started.date())
    while day >= earliest:
        rows.append([day.isoformat(), data["days"].get(day.isoformat(), {}).get(channel, 0)])
        day -= timedelta(days=1)
    summary = (f"**Last 30 days ({first_day:%Y-%m-%d} – {local:%Y-%m-%d}):** "
               f"{total:,} successful runs.\n\n"
               f"The total includes today and the preceding 29 days, using {TIMEZONE}. "
               f"Tracking started {started:%Y-%m-%d %H:%M %Z}. "
               "Earlier activity is not included. Only completed enhancement or generation runs count.")
    return summary, rows


def success_badges(data, channel, *, now=None, title=None):
    """Keep unavailable history distinct from a measured count of zero."""
    local, first_day, started, total = rolling_period(data, channel, now=now)
    recent, today = "—", "—"
    recent_note = "No historical data"
    today_note = "Success counts are temporarily unavailable"
    if started is not None:
        recent = f"{total:,}"
        today = f'{data["days"].get(local.date().isoformat(), {}).get(channel, 0):,}'
        today_note = f"Successful runs on {local:%Y-%m-%d} ({TIMEZONE})"
        recent_note = (f"Successful runs from {first_day:%Y-%m-%d} through {local:%Y-%m-%d} "
                       f"({TIMEZONE}); recorded since {started:%Y-%m-%d}")

    def chip(label, value, note):
        return (
            '<span style="display:inline-flex;align-items:center;gap:8px;padding:7px 12px;'
            'border:1px solid var(--border-color-primary,#d1d5db);border-radius:999px;'
            'background:var(--background-fill-secondary,#f9fafb);font-size:14px;" '
            f'title="{html.escape(note, quote=True)}">'
            f'<span style="color:var(--body-text-color-subdued,#6b7280)">{label}</span>'
            f'<strong style="font-variant-numeric:tabular-nums">{html.escape(value)}</strong></span>'
        )

    badges = ('<div style="display:flex;flex-wrap:wrap;justify-content:flex-end;gap:10px;'
            'margin-left:auto;flex:0 0 auto;" '
            'aria-label="Successful runs">'
            + chip("Last 30 days", recent, recent_note)
            + chip("Today", today, today_note) + '</div>')
    if title is None:
        return badges
    return ('<div style="display:flex;align-items:center;justify-content:space-between;'
            'flex-wrap:wrap;gap:12px;width:100%;margin:0 0 8px;">'
            '<h1 style="flex:1 1 20rem;min-width:0;margin:0;padding:0;'
            'font-size:clamp(1.4rem,2.5vw,2rem);font-weight:600;line-height:1.25;">'
            + html.escape(title) + '</h1>' + badges + '</div>')


class UsageDisplay:
    """Align right-side badges with the title and refresh after successful runs."""

    def __init__(self, stats, channel):
        self.stats = stats
        self.channel = channel
        self.header = None
        self.title = None

    def render_header(self, title):
        import gradio as gr
        self.title = title
        self.header = gr.HTML(success_badges(None, self.channel, title=title), elem_classes=["success-counts"])

    def load(self):
        try:
            data = self.stats.snapshot()
        except Exception:
            return success_badges(None, self.channel, title=self.title), "Success counts are temporarily unavailable.", []
        summary, rows = summarize(data, self.channel)
        note = summary.split("\n\n", 1)[-1]
        return success_badges(data, self.channel, title=self.title), note, rows

    def bind(self, demo, generation):
        import gradio as gr
        if self.header is None:
            raise ValueError("Render the usage header before binding its events")
        with gr.Accordion("Daily success history", open=False):
            summary = gr.Markdown("Loading success counts…")
            daily = gr.Dataframe(headers=[f"Date ({TIMEZONE})", "Successful runs"],
                                 datatype=["str", "number"], value=[], interactive=False,
                                 label="Daily successful runs", row_count=(0, "dynamic"))
            refresh = gr.Button("Refresh counts", size="sm")
        outputs = [self.header, summary, daily]

        def refresh_counts():
            with self.stats.lock:
                self.stats.cached = None
            return self.load()

        demo.load(self.load, outputs=outputs, queue=False, api_name=False)
        refresh.click(refresh_counts, outputs=outputs, queue=False, api_name=False)
        generation.success(self.load, outputs=outputs, queue=False, api_name=False)
