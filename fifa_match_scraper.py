#python fifa_match_scraper.py --date-from 2024-01-01 --date-till 2024-01-31 -o matches.csv

from __future__ import annotations

import argparse
import asyncio
import csv
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
import time
from playwright.async_api import Browser, Locator, Page, TimeoutError as PlaywrightTimeoutError, async_playwright

URL = "https://inside.fifa.com/data-centre/matches/men"
DATE_FORMATS = ("%d %b %Y", "%d %B %Y")
DATE_RE = re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b")
SCORE_TOKEN_RE = re.compile(r"^\(?\d+\)?$")
STATUS_RE = re.compile(r"^(?:FT|AET|PEN|AP|LIVE|HT|POSTPONED|CANCELLED|ABANDONED)$", re.I)


@dataclass(frozen=True)
class Match:
    date: str
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    home_penalty_score: Optional[int]
    away_penalty_score: Optional[int]
    tournament: str
    stage: str
    stadium: str


def parse_cli_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dates must use YYYY-MM-DD") from exc


def parse_display_date(value: str) -> date:
    value = re.sub(r"\s+", " ", value.strip())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized FIFA date: {value!r}")


def clean_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if x.strip()]


def parse_score_tokens(tokens: list[str]) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return home, away, home_penalty, away_penalty.

    FIFA visually renders shootouts like `(4) 0 (3) 0`. This function also accepts
    ordinary scores such as `2 1` and several alternative token orders.
    """
    numeric = [t for t in tokens if SCORE_TOKEN_RE.fullmatch(t)]
    penalties = [int(t[1:-1]) for t in numeric if t.startswith("(")]
    regulation = [int(t) for t in numeric if not t.startswith("(")]

    home = regulation[0] if len(regulation) >= 1 else None
    away = regulation[1] if len(regulation) >= 2 else None
    home_pen = penalties[0] if len(penalties) >= 1 else None
    away_pen = penalties[1] if len(penalties) >= 2 else None
    return home, away, home_pen, away_pen


async def accept_cookies(page: Page) -> None:
    candidates = [
        page.get_by_role("button", name=re.compile(r"Reject All", re.I)),
    ]
    for locator in candidates:
        try:
            if await locator.first.is_visible(timeout=1500):
                await locator.first.click()
                return
        except PlaywrightTimeoutError:
            continue


async def match_row_locators(page: Page) -> Locator:
    # Prefer true table rows. Fall back to elements whose text begins with a date.
    rows = page.locator("table tbody tr")
    if await rows.count():
        return rows

    # Playwright applies the regular expression to element text. This fallback may
    # include nested containers; duplicate records are removed after parsing.
    return page.locator("article, li, div").filter(has_text=DATE_RE)


async def visible_match_texts(page: Page) -> list[str]:
    rows = await match_row_locators(page)
    texts: list[str] = []
    for i in range(await rows.count()):
        text = (await rows.nth(i).inner_text()).strip()
        if DATE_RE.search(text):
            texts.append(text)
    return texts


async def oldest_loaded_date(page: Page) -> Optional[date]:
    dates: list[date] = []
    for text in await visible_match_texts(page):
        m = DATE_RE.search(text)
        if m:
            try:
                dates.append(parse_display_date(m.group(1)))
            except ValueError:
                pass
    return min(dates) if dates else None


async def click_show_more_until(page: Page, target: date, max_clicks: int) -> None:
    """Click until at least one match on/before target is loaded."""
    previous_oldest: Optional[date] = None

    for click_number in range(max_clicks + 1):
        oldest = await oldest_loaded_date(page)
        if oldest is not None:
            print(f"Loaded through {oldest.isoformat()}")
            if oldest <= target:
                return

        if click_number == max_clicks:
            raise RuntimeError(f"Reached --max-clicks={max_clicks} before loading {target.isoformat()}")
        time.sleep(5)
        button = page.get_by_role("button", name=re.compile(r"Show more", re.I)).last
        try:
            await button.wait_for(state="visible", timeout=10_000)
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"'Show more' disappeared before matches from {target.isoformat()} were loaded"
            ) from exc

        before_count = len(await visible_match_texts(page))
        await button.scroll_into_view_if_needed()
        await button.click()

        # Wait for more rows or for the oldest date to change.
        for _ in range(40):

            await page.wait_for_timeout(250)
            time.sleep(3)
            now_count = len(await visible_match_texts(page))
            now_oldest = await oldest_loaded_date(page)
            if now_count > before_count or (now_oldest and now_oldest != oldest):
                break
        else:
            raise RuntimeError("Clicking 'Show more' did not load additional match rows")

        if oldest is not None and previous_oldest == oldest:
            raise RuntimeError("The oldest loaded match date stopped changing")
        previous_oldest = oldest


async def structured_row_values(row: Locator) -> list[str]:
    """Extract cell-like values while ignoring nested duplicate text."""
    cells = row.locator(":scope > th, :scope > td")
    if await cells.count():
        return [re.sub(r"\s+", " ", (await cells.nth(i).inner_text()).strip()) for i in range(await cells.count())]
    return clean_lines(await row.inner_text())


def parse_row(values: list[str]) -> Optional[Match]:
    # Split any multiline/combined cells once more.
    items = [part for value in values for part in clean_lines(value)]
    if not items:
        return None

    date_index = next((i for i, x in enumerate(items) if DATE_RE.fullmatch(x)), None)
    if date_index is None:
        # A date may share a cell with another field.
        joined = "\n".join(items)
        match = DATE_RE.search(joined)
        if not match:
            return None
        display_date = match.group(1)
    else:
        display_date = items[date_index]
        items = items[date_index:]

    # Remove labels/header words if they leaked into the row.
    ignored = {"date", "match result", "tournament", "stage", "stadium"}
    items = [x for x in items if x.casefold() not in ignored]
    if items and DATE_RE.fullmatch(items[0]):
        items = items[1:]

    # Status sits between team names and score on the current site.
    status_idx = next((i for i, x in enumerate(items) if STATUS_RE.fullmatch(x)), None)
    if status_idx is None or status_idx < 2:
        raise ValueError(f"Could not identify teams/status in row: {values!r}")

    home_team, away_team = items[status_idx - 2], items[status_idx - 1]
    after_status = items[status_idx + 1:]

    score_tokens: list[str] = []
    remainder_start = 0
    for i, token in enumerate(after_status):
        if SCORE_TOKEN_RE.fullmatch(token):
            score_tokens.append(token)
            remainder_start = i + 1
        elif score_tokens:
            break

    # Some layouts put score before FT. Handle that as a fallback.
    if len([x for x in score_tokens if not x.startswith("(")]) < 2:
        preceding = items[:status_idx - 2]
        score_tokens = [x for x in preceding if SCORE_TOKEN_RE.fullmatch(x)]
        remainder = after_status
    else:
        remainder = after_status[remainder_start:]

    home_score, away_score, home_pen, away_pen = parse_score_tokens(score_tokens)

    # The final three semantic columns are tournament, stage, stadium. Empty stage
    # may collapse in text extraction, so preserve the best possible assignment.
    if len(remainder) >= 3:
        tournament = remainder[0]
        stage = remainder[1]
        stadium = " ".join(remainder[2:])
    elif len(remainder) == 2:
        tournament, stadium = remainder
        stage = ""
    elif len(remainder) == 1:
        tournament, stage, stadium = remainder[0], "", ""
    else:
        tournament = stage = stadium = ""

    return Match(
        date=parse_display_date(display_date).isoformat(),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        home_penalty_score=home_pen,
        away_penalty_score=away_pen,
        tournament=tournament,
        stage=stage,
        stadium=stadium,
    )


async def scrape(page: Page, date_from: date, date_till: date, max_clicks: int) -> list[Match]:
    await page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
    await accept_cookies(page)

    # Wait for the archive, not merely the shell HTML.
    await page.get_by_text("International match archive", exact=True).first.wait_for(timeout=60_000)
    await page.wait_for_timeout(1500)
    time.sleep(3)
    await click_show_more_until(page, date_from - timedelta(days=1), max_clicks)

    rows = await match_row_locators(page)
    matches: list[Match] = []
    errors: list[str] = []

    for i in range(await rows.count()):
        values = await structured_row_values(rows.nth(i))
        try:
            match = parse_row(values)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if match is None:
            continue
        match_date = date.fromisoformat(match.date)
        if date_from <= match_date <= date_till:
            matches.append(match)

    # Remove duplicate DOM representations while preserving page order.
    unique: dict[tuple, Match] = {}
    for m in matches:
        key = (
            m.date, m.home_team, m.away_team, m.home_score, m.away_score,
            m.home_penalty_score, m.away_penalty_score, m.tournament, m.stage, m.stadium,
        )
        unique.setdefault(key, m)

    result = sorted(unique.values(), key=lambda m: (m.date, m.home_team, m.away_team))
    if not result:
        sample = "\n".join(errors[:3])
        raise RuntimeError(
            "No matches were parsed in the requested range. FIFA may have changed its markup."
            + (f"\nParser samples:\n{sample}" if sample else "")
        )
    return result


def write_csv(path: Path, matches: Iterable[Match]) -> None:
    rows = [asdict(m) for m in matches]
    fieldnames = list(Match.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def async_main(args: argparse.Namespace) -> None:
    if args.date_from > args.date_till:
        raise SystemExit("--date-from must be on or before --date-till")

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(headless=not args.headed)
        context = await browser.new_context(
            locale="en-GB",
            viewport={"width": 1440, "height": 1200},
        )
        page = await context.new_page()
        try:
            matches = await scrape(page, args.date_from, args.date_till, args.max_clicks)
        finally:
            await browser.close()

    write_csv(args.output, matches)
    print(f"Wrote {len(matches)} matches to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-from", required=True, type=parse_cli_date, help="inclusive YYYY-MM-DD")
    parser.add_argument("--date-till", required=True, type=parse_cli_date, help="inclusive YYYY-MM-DD")
    parser.add_argument("-o", "--output", type=Path, default=Path("fifa_matches.csv"))
    parser.add_argument("--max-clicks", type=int, default=500, help="safety limit for Show more clicks")
    parser.add_argument("--headed", action="store_true", help="show Chromium for debugging")
    return parser


if __name__ == "__main__":
    asyncio.run(async_main(build_parser().parse_args()))
