"""Minimal, dependency-free Praat TextGrid parsing and activity utilities.

AliMeeting TextGrids use the long text format.  We deliberately parse only
IntervalTier entries with non-empty text: those entries are the speech turns.
The parser is encoding tolerant because some delivered transcript text renders
as mojibake on non-Chinese Windows locales; interval boundaries and tier names
remain ASCII and are unaffected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True, order=True)
class ActivityRegion:
    start: float
    end: float
    speakers: tuple[str, ...]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


_NAME_RE = re.compile(r'^\s*name\s*=\s*"(.*)"\s*$')
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\d+\]\s*:")
_XMIN_RE = re.compile(r"^\s*xmin\s*=\s*([-+0-9.eE]+)")
_XMAX_RE = re.compile(r"^\s*xmax\s*=\s*([-+0-9.eE]+)")
_TEXT_RE = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')


def parse_textgrid(path: str | Path) -> dict[str, list[Segment]]:
    """Return non-empty speech intervals grouped by tier name."""

    tiers: dict[str, list[Segment]] = defaultdict(list)
    current_tier: str | None = None
    xmin: float | None = None
    xmax: float | None = None
    in_interval = False

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _NAME_RE.match(line)
            if match:
                current_tier = match.group(1)
                xmin = xmax = None
                in_interval = False
                continue

            if _INTERVAL_RE.match(line):
                xmin = xmax = None
                in_interval = True
                continue

            if not in_interval or current_tier is None:
                continue

            match = _XMIN_RE.match(line)
            if match:
                xmin = float(match.group(1))
                continue
            match = _XMAX_RE.match(line)
            if match:
                xmax = float(match.group(1))
                continue
            match = _TEXT_RE.match(line)
            if match and xmin is not None and xmax is not None:
                text = match.group(1).strip()
                if text and xmax > xmin:
                    tiers[current_tier].append(
                        Segment(float(xmin), float(xmax), current_tier, text)
                    )
                xmin = xmax = None
                in_interval = False

    return dict(tiers)


def flatten_segments(
    tiers: Mapping[str, Iterable[Segment]],
) -> list[Segment]:
    return sorted(segment for segments in tiers.values() for segment in segments)


def activity_regions(
    tiers: Mapping[str, Iterable[Segment]],
    *,
    start: float = 0.0,
    end: float | None = None,
) -> list[ActivityRegion]:
    """Convert possibly overlapping turns into disjoint constant-activity regions.

    Counts are used rather than a plain set, so malformed overlapping intervals
    from the same speaker cannot prematurely deactivate that speaker.
    """

    events: dict[float, list[tuple[str, int]]] = defaultdict(list)
    maximum = start
    for speaker, segments in tiers.items():
        for segment in segments:
            left = max(start, float(segment.start))
            right = float(segment.end) if end is None else min(end, float(segment.end))
            if right <= left:
                continue
            events[left].append((speaker, +1))
            events[right].append((speaker, -1))
            maximum = max(maximum, right)

    terminal = maximum if end is None else end
    if terminal <= start:
        return []
    events.setdefault(start, [])
    events.setdefault(terminal, [])

    active: dict[str, int] = defaultdict(int)
    output: list[ActivityRegion] = []
    previous = min(events)
    for time in sorted(events):
        if time > previous:
            speakers = tuple(sorted(s for s, count in active.items() if count > 0))
            if speakers:
                output.append(ActivityRegion(previous, time, speakers))
        # All state changes at a timestamp happen after scoring the preceding span.
        for speaker, delta in events[time]:
            active[speaker] += delta
        previous = time

    return output


def intersect_duration(
    regions: Iterable[ActivityRegion],
    left: float,
    right: float,
    predicate,
) -> float:
    """Duration inside [left, right) for regions satisfying ``predicate``."""

    total = 0.0
    for region in regions:
        if region.end <= left:
            continue
        if region.start >= right:
            break
        if predicate(region):
            total += max(0.0, min(right, region.end) - max(left, region.start))
    return total


def intervals_for_speaker(
    tiers: Mapping[str, Iterable[Segment]], speaker: str
) -> list[tuple[float, float]]:
    return [(segment.start, segment.end) for segment in tiers.get(speaker, [])]

