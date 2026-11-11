# Successful-run counts

The hosted demo records one successful run after image enhancement and output
brightness processing finish. Invalid inputs, failed inference, brightness-only
adjustments, page visits, and downloads do not increment the counter.

Badges below the page title show the previous calendar month's total and today's
successful runs. The **Daily success history** panel contains the daily table
for the previous and current months. Dates use
`America/Los_Angeles`. Missing history is displayed as **No historical data**;
the first incomplete month is explicitly marked as partial.

Counters are stored in the private dataset `CaiYuanhao/space-success-stats`.
Set `SPACE_STATS_TOKEN` as a Space Secret with a long-lived fine-grained token
that can read and write only that dataset. `SPACE_STATS_REPO` optionally selects
another initialized private dataset with the same schema. Tokens must never be
committed to the repository. The interactive login's expiring OAuth token is not
used as a runtime secret.

If the Secret has another name, set the non-sensitive `SPACE_STATS_TOKEN_NAME`
Variable to that name. Existing deployments use `cyh_track_liuliang`.

Only dates, per-demo counts, instrumentation start times, and random event IDs
for retry deduplication are stored. No images or user identifiers are collected.
Optimistic commits prevent concurrent writers from overwriting one another.
If storage remains unavailable after retries, inference still returns its result
and an anonymous `SUCCESS_COUNT_NOT_SAVED` entry is logged for recovery.

Badges load on page opening and update automatically after successful processing.
**Refresh counts** reloads the badges and history from storage.
Counts are cached for up to 30 seconds across processes.

Run the persistence and calendar-boundary tests without GPU dependencies:

```sh
python -m unittest discover -s tests -p test_space_success_stats.py -v
```
