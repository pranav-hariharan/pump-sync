"""Evaluation of PumpSync playlists against simple baselines."""

import numpy as np
import pandas as pd

from src.data import FEATURES
from src.matching import GYM_EXCLUDE_GENRES, PHASE_SHARES, build_playlist

MIN_POPULARITY = 40  # matches the default in top_matches


def baseline_playlist(
    lib,
    duration_min=45,
    mode="random",
    include_genres=None,
    min_popularity=MIN_POPULARITY,
    max_per_artist=2,
    seed=None,
):
    """Build a playlist without audio matching, from the same candidate pool.

    mode="random": random tracks in random order.
    mode="popularity": the most popular tracks, most popular first.
    Phases are assigned by position in time using PHASE_SHARES.
    """
    df = lib.df
    mask = (df["popularity"] >= min_popularity) & ~df["track_genre"].isin(
        GYM_EXCLUDE_GENRES
    )
    if include_genres:
        mask &= df["track_genre"].isin(include_genres)
    pool = df[mask]
    pool = (
        pool.sample(frac=1, random_state=seed)
        if mode == "random"
        else pool.sort_values("popularity", ascending=False)
    )

    budget = duration_min * 60_000
    chosen, total, artist_counts = [], 0, {}
    for _, row in pool.iterrows():
        if total >= budget:
            break
        artist = row["main_artist"]
        if artist_counts.get(artist, 0) >= max_per_artist:
            continue
        chosen.append(row)
        total += row["duration_ms"]
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

    out = pd.DataFrame(chosen, columns=pool.columns).reset_index(drop=True)
    start_share = (out["duration_ms"].cumsum() - out["duration_ms"]) / out[
        "duration_ms"
    ].sum()
    warmup_end = PHASE_SHARES["warmup"]
    peak_end = warmup_end + PHASE_SHARES["peak"]
    out["phase"] = np.select(
        [start_share < warmup_end, start_share < peak_end],
        ["warmup", "peak"],
        "cooldown",
    )

    # Same ordering rule PumpSync uses: warmup rising, cooldown falling, peak untouched
    warmup = out[out["phase"] == "warmup"].sort_values("energy", ascending=True)
    peak = out[out["phase"] == "peak"]
    cooldown = out[out["phase"] == "cooldown"].sort_values("energy", ascending=False)
    return pd.concat([warmup, peak, cooldown]).reset_index(drop=True)


def score_playlist(lib, playlist, plan):
    """Return evaluation metrics for one playlist."""
    scaled = lib.scaler.transform(playlist[FEATURES])
    dists = []
    for phase in ["warmup", "peak", "cooldown"]:
        rows = (playlist["phase"] == phase).to_numpy()
        if rows.any():
            target = lib.scaler.transform(pd.DataFrame([plan[phase]])[FEATURES])
            dists.extend(np.linalg.norm(scaled[rows] - target, axis=1))

    by_phase = playlist.groupby("phase")["energy"].mean()
    return {
        "target_distance": float(np.mean(dists)),
        "energy_jump": float(np.abs(np.diff(playlist["energy"].to_numpy())).mean()),
        "arc_gap": float(
            by_phase.get("peak", np.nan) - by_phase.get("cooldown", np.nan)
        ),
        "minutes": playlist["duration_ms"].sum() / 60_000,
    }


def run_evaluation(lib, parser, cases, seeds=(1, 2, 3), duration_min=45):
    """Score every method on every (prompt, genre filter) case and seed."""
    rows = []
    for text, genres in cases:
        plan, used_fallback = parser.parse(text)
        if used_fallback:
            print("skipped (LLM call failed):", text)
            continue
        for seed in seeds:
            methods = {
                "random": baseline_playlist(
                    lib, duration_min, "random", genres, seed=seed
                ),
                "popularity": baseline_playlist(
                    lib, duration_min, "popularity", genres, seed=seed
                ),
                "pumpsync": build_playlist(
                    lib, plan, duration_min, seed=seed, include_genres=genres
                ),
                "pumpsync_phase_clusters": build_playlist(
                    lib,
                    plan,
                    duration_min,
                    seed=seed,
                    include_genres=genres,
                    restrict_to_phase_clusters=True,
                ),
            }
            for name, playlist in methods.items():
                rows.append(
                    {
                        "method": name,
                        "prompt": text,
                        "genres": str(genres),
                        "seed": seed,
                        **score_playlist(lib, playlist, plan),
                    }
                )
    return pd.DataFrame(rows)
