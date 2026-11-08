import random

import streamlit as st

from src.data import load_prepared
from src.matching import Library, build_playlist
from src.mood import MoodParser

st.set_page_config(page_title="PumpSync")


@st.cache_resource
def get_resources():
    # cached so the data loads once per server process, not on every click
    df, X, scaler = load_prepared()
    lib = Library(df, X, scaler)
    return lib, MoodParser(lib)


lib, parser = get_resources()

st.title("PumpSync")
st.write("Describe your workout and get a warmup, peak and cooldown playlist.")

text = st.text_area(
    "Session description",
    placeholder="e.g. heavy leg day, feeling tired",
    max_chars=300,
)
duration = st.slider("Length (minutes)", 30, 60, 45, step=5)
genres = st.multiselect(
    "Genres (leave empty for all)",
    sorted(lib.df["track_genre"].unique()),
)

if st.button("Generate playlist", type="primary", disabled=not text.strip()):
    plan, used_fallback = parser.parse(text)
    # new random seed each click, so clicking again gives a different playlist
    playlist = build_playlist(
        lib,
        plan,
        duration,
        seed=random.randrange(10**6),
        include_genres=genres or None,
    )

    if used_fallback:
        st.warning(
            "Gemini failed, so a default plan was used instead of your description."
        )

    total_min = playlist["duration_ms"].sum() / 60_000
    st.subheader(f"{len(playlist)} tracks, {total_min:.0f} min")

    table = playlist.assign(
        length=(playlist["duration_ms"] // 1000).map(
            lambda s: f"{s // 60}:{s % 60:02d}"
        ),
        link="https://open.spotify.com/track/" + playlist["track_id"],
    )[["phase", "track_name", "artists", "length", "link"]]

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn("Spotify", display_text="Open")
        },
    )

    st.write("Energy across the playlist")
    st.line_chart(playlist["energy"].reset_index(drop=True))
