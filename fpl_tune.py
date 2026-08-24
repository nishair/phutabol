#!/usr/bin/env python3
"""
Tune the SeasonManager across many historical seasons.

Backtests every season the archive supports (2017/18 onward — each needs
its prior season for projections), runs a grid of ManagerConfig variants
over all of them in parallel, and ranks variants by mean season score.
Tuning on many seasons guards against overfitting any single one.

Notes on old data:
- xGI exists from 2022/23, so the luck regression is inactive before
  2023/24 backtests.
- birth dates only appear in recent files; they are time-invariant, so
  they are backfilled to older seasons by permanent player code.
- 2019/20 ran to GW47 (COVID restart); chip windows scale with season
  length.

Usage:
    python fpl_tune.py [--seasons 2017-18 ... ] [--jobs 6]
"""

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from phutabol.fpl import build_projections
from phutabol.fpl.season import ManagerConfig, SeasonData, SeasonManager
from fpl_backtest import (
    ARCHIVE, build_season_data, build_season_start_bootstrap,
    previous_season,
)

CACHE = Path(".fpl_cache")

DEFAULT_SEASONS = [
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]

# Seasons whose players_raw carries birth_date, newest first.
BIRTH_DATE_SEASONS = ["2025-26", "2024-25"]

VARIANTS: Dict[str, ManagerConfig] = {
    "base": ManagerConfig(),
    "fast_learn": ManagerConfig(blend_minutes=450, avail_prior=45),
    "slow_learn": ManagerConfig(blend_minutes=1400),
    "active": ManagerConfig(
        ft_gain=2.5, hit_gain=8.0, max_hits_per_week=2
    ),
    "fast_active": ManagerConfig(
        blend_minutes=450, avail_prior=45,
        ft_gain=2.5, hit_gain=8.0, max_hits_per_week=2,
    ),
    "cheap_bench": ManagerConfig(
        blend_minutes=450, avail_prior=45,
        ft_gain=2.5, hit_gain=8.0, max_hits_per_week=2,
        bench_weight=0.08,
    ),
    "chip_eager": ManagerConfig(
        blend_minutes=450, avail_prior=45,
        ft_gain=2.5, hit_gain=8.0, max_hits_per_week=2,
        wildcard_gain=10.0, free_hit_gain=9.0,
        bench_boost_min=12.0, triple_captain_min=9.0,
    ),
    "chip_patient": ManagerConfig(
        blend_minutes=450, avail_prior=45,
        ft_gain=2.5, hit_gain=8.0, max_hits_per_week=2,
        wildcard_gain=20.0, free_hit_gain=16.0,
        bench_boost_min=18.0, triple_captain_min=14.0,
    ),
}


def fetch_cached(path: str) -> Optional[pd.DataFrame]:
    """Fetch an archive CSV through a local cache; None if missing."""
    CACHE.mkdir(exist_ok=True)
    local = CACHE / path.replace("/", "_")
    if not local.exists():
        response = requests.get(
            f"{ARCHIVE}/{path}", timeout=60,
            headers={"User-Agent": "Mozilla/5.0 (phutabol)"},
        )
        if response.status_code != 200:
            return None
        local.write_bytes(response.content)
    raw = local.read_bytes()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:  # 2016-2019 files are latin-1
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1")


def birth_date_map() -> Dict[int, str]:
    """Permanent player code -> ISO birth date, from recent seasons."""
    dates: Dict[int, str] = {}
    for season in BIRTH_DATE_SEASONS:
        players = fetch_cached(f"{season}/players_raw.csv")
        if players is None or "birth_date" not in players.columns:
            continue
        for row in players.itertuples():
            if pd.notna(row.birth_date):
                dates.setdefault(row.code, str(row.birth_date))
    return dates


def prepare_season(
    season: str, birth_dates: Dict[int, str]
) -> Optional[Tuple[list, SeasonData]]:
    """Build (projections, SeasonData) for one season, or None."""
    players = fetch_cached(f"{season}/players_raw.csv")
    prior = fetch_cached(f"{previous_season(season)}/players_raw.csv")
    gws = fetch_cached(f"{season}/gws/merged_gw.csv")
    fixtures = fetch_cached(f"{season}/fixtures.csv")
    if any(df is None for df in (players, prior, gws, fixtures)):
        return None

    teams = fetch_cached(f"{season}/teams.csv")
    if teams is None:  # cosmetic only: short_name for display
        teams = pd.DataFrame({
            "id": sorted(players["team"].unique()),
        })
        teams["short_name"] = teams["id"].map("T{}".format)

    bootstrap = build_season_start_bootstrap(
        players, prior, teams, birth_dates=birth_dates
    )
    projections = build_projections(
        bootstrap, fixtures.to_dict("records"), next_event=1,
        as_of=date(int(season.split("-")[0]), 8, 1),
    )
    start_prices = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
    data = build_season_data(gws, fixtures, start_prices)
    return projections, data


def run_one(task) -> Tuple[str, str, int]:
    season, variant, projections, data, config = task
    manager = SeasonManager(projections, data, config=config)
    weeks = manager.run()
    return season, variant, sum(w.points for w in weeks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the season manager")
    parser.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    birth_dates = birth_date_map()
    bundles = {}
    for season in args.seasons:
        bundle = prepare_season(season, birth_dates)
        if bundle is None:
            print(f"  {season}: archive incomplete, skipped")
            continue
        bundles[season] = bundle
    print(f"Tuning over {len(bundles)} seasons × {len(VARIANTS)} variants…")

    tasks = [
        (season, variant, projections, data, config)
        for season, (projections, data) in bundles.items()
        for variant, config in VARIANTS.items()
    ]
    results: Dict[str, Dict[str, int]] = {v: {} for v in VARIANTS}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for season, variant, total in pool.map(run_one, tasks):
            results[variant][season] = total

    seasons = sorted(bundles)
    header = "variant".ljust(14) + "".join(s[2:7].center(9) for s in seasons)
    print("\n" + header + "   mean".rjust(8))
    print("-" * len(header))
    ranked = sorted(
        results.items(),
        key=lambda kv: -sum(kv[1].values()) / len(kv[1]),
    )
    for variant, by_season in ranked:
        mean = sum(by_season.values()) / len(by_season)
        row = "".join(str(by_season[s]).center(9) for s in seasons)
        print(variant.ljust(14) + row + f"{mean:8.0f}")

    best = ranked[0][0]
    print(f"\nBest by mean: {best}\n{VARIANTS[best]}")


if __name__ == "__main__":
    main()
