"""Data loading, cleaning, scaling and clustering for PumpSync"""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "spotify-tracks-dataset.csv"
)

# Features to use for clustering
FEATURES = [
    "energy",
    "valence",
    "tempo",
    "danceability",
    "loudness",
    "acousticness",
    "instrumentalness",
]

# Genres that do not suit a workout playlist
EXCLUDE_GENRES = [
    "comedy",
    "children",
    "kids",
    "disney",
    "sleep",
    "study",
    "opera",
    "new-age",
    "show-tunes",
    "romance",
    "sad",
]

N_CLUSTERS = 5  # Chosen from elbow + silhouette scores

# Manually labeled cluster names, based on the mean feature values of each K-means cluster
CLUSTER_NAMES = {
    0: "calm-instrumentals",
    1: "upbeat",
    2: "hype-aggressive",
    3: "chill-laidback",
    4: "intense",
}

# Which clusters can fill each workout phase (cluster 0 is excluded since it is not suitable for gym music)
PHASE_CLUSTERS = {
    "warmup": [1],
    "peak": [2, 4],
    "cooldown": [3],
}


def load_and_clean(path=DATA_PATH):
    """Load the CSV and clean the data."""
    df = pd.read_csv(path)

    # Drop leftover index columns from the CSV export
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df = df.dropna()
    df = df[~df["track_genre"].isin(EXCLUDE_GENRES)]

    # Same song appears once per genre tag, so keep one row per track_id
    df = df.drop_duplicates(subset="track_id")

    # Same song also appears under different track_ids (remixes).
    # Keep the most popular version of each title + main artist.
    df["main_artist"] = df["artists"].str.split(";").str[0]
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "main_artist"])

    df = df[
        (df["tempo"] > 0)  # 0 BPM = junk data
        & (
            df["duration_ms"].between(120_000, 420_000)
        )  # 2-7 mins keeps length appropriate for workout
        & (df["speechiness"] < 0.66)
    ]  # Doesn't suit a workout

    return df.reset_index(drop=True)


def build_model(df):
    """Scale the features and cluster the songs through K-means."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    df = df.copy()
    df["cluster"] = km.fit_predict(X)
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)
    return df, X, scaler, km


def check_clusters(df):
    """Raise AssertionError if the cluster numbering no longer matches CLUSTER_NAMES."""
    p = df.groupby("cluster")[FEATURES].mean()
    assert p.loc[0, "energy"] < 0.3, "cluster 0 should be quiet"
    assert p.loc[2, "tempo"] > p.loc[3, "tempo"] and p.loc[2, "energy"] > 0.75, (
        "cluster 2 should be hype-aggressive"
    )
    assert p.loc[3, "energy"] < 0.5, "cluster 3 should be chill"


def load_all():
    """Load, clean, scale and cluster the data."""
    df = load_and_clean()
    df, X, scaler, km = build_model(df)
    check_clusters(df)
    return df, X, scaler, km
