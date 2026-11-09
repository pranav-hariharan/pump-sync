# PumpSync

Describe your workout in plain text and get a playlist shaped like a gym session: warmup, peak, cooldown.

**Live demo:** https://pumpsync.streamlit.app

Example input: `heavy leg day, feeling tired`. Output: a ~45 minute playlist (length adjustable, 30-60 min) with Spotify links, split 20% warmup / 60% peak / 20% cooldown, with an energy curve.

## How it works

1. **Text to targets.** The Gemini API turns the description into target audio features (energy, valence, tempo, danceability, loudness, acousticness, instrumentalness) for each phase. It returns schema-validated JSON (Pydantic). The prompt includes each phase's median feature profile from the data, and the model may only shift values a limited amount from that profile. Returned targets are clamped to the 5th-95th percentile range of that phase's tracks. If every model fails, a hand-written default plan is used and the app shows a warning.
2. **Targets to tracks.** Features are standardized (StandardScaler) and each phase's target is scaled with the same scaler. The closest tracks by Euclidean distance form a candidate pool, and tracks are sampled randomly from that pool, so regenerating gives a different playlist.
3. **Playlist assembly.** Each phase is filled to its share of the requested length. Max 2 tracks per artist, no repeats. Warmup is ordered by rising energy, cooldown by falling energy, peak is left shuffled.

The LLM only translates text into numbers. Data cleaning, clustering, matching, ordering and evaluation are my own code.

## Data

- **Source:** "490K Spotify Song Audio Embeddings & Metadata" by serkantysz (Kaggle), 491,632 tracks. Spotify's audio-features endpoint has been closed to new apps since Nov 2024, so all audio data comes from this file.
- **Cleaning (491,632 to 155,036 tracks):**
  - Release year 1990 or later, and popularity of at least 20, to drop obscure uploads.
  - Removed slowed, sped-up, nightcore and live versions, which are alternate takes of other songs.
  - Deduplicated by normalized title + main artist (remaster/radio edit/mono/stereo suffixes stripped), keeping the most popular version.
  - Removed invalid tempo (0), tracks outside 2-7 minutes, and spoken-word tracks (speechiness 0.66 or higher).
- **Not used:** the lyrics column and the audio embedding files.
- The deployed app loads `data/processed/tracks_prepared.csv.gz` (10.8 MB: cleaned data, needed columns, cluster labels), not the 770 MB raw CSV.

## Design decisions

- **Euclidean distance, not cosine.** Cosine ignores magnitude, but "how energetic" is a magnitude. I never ran a cosine comparison, so this is a reasoned choice, not a measured one.
- **The LLM outputs numbers, not embeddings.** Text embeddings and audio features live in different spaces and can't be compared directly.
- **Standardize first.** Without it, tempo (about 60-200) and loudness (negative dB) would dominate distances over 0-1 features. The scaler is kept so LLM targets get the same transform.
- **K-Means, k=5.** Elbow was around 4-5 and the silhouette gap between k=4 and k=5 was within noise, so this was a judgment call. k=5 separates a fast (about 160 BPM) cluster from a hard mid-tempo (about 108 BPM) one. Silhouette is about 0.2, so clusters overlap heavily: music is more of a continuum than five groups.
- **Cluster-to-phase mapping is hand-made.** Cluster numbers are arbitrary and reshuffle when the data changes. `check_clusters()` asserts mean energy is peak > warmup > cooldown and caught one reshuffle.

## Evaluation

20 prompt/genre cases (19 distinct prompts) x 3 seeds, compared with two baselines drawing from the same candidate pool. Baselines are ordered by energy within phases the same way PumpSync is.

| Method | Distance to phase targets (lower is better) | Mean energy change between tracks | Peak minus cooldown energy | Minutes |
|---|---|---|---|---|
| Random | 2.69 | 0.19 | 0.06 | 47.0 |
| Most popular | 2.84 | 0.20 | -0.13 | 46.8 |
| **PumpSync** | **0.67** | **0.10** | **0.54** | 45.7 |

PumpSync was closer to the targets than random in 100% of cases.

**Caveats:**
- The targets are PumpSync's own, so distance to them and the energy arc are partly circular. This shows the matching does what it was built to do, not that the playlists sound good.
- No metric here measures taste. A blind listening test would; I haven't run one.
- Results come from one dataset, 19 prompts and 3 seeds.

**K-Means did not improve matching.** Restricting matching to the clusters assigned to each phase made no difference to distance (PumpSync with and without it: 0.67 vs 0.67). K-Means contributes the phase profiles used in the prompt and clamp, and the analysis (e.g. hip-hop mostly lives in the groovy-danceable cluster).

## Limitations

- The dataset thins out after about 2023, so recent tracks are missing.
- Audio features can't tell a love song from a gym track. Genre filtering helps, but genre tags are coarse and noisy (Rage Against the Machine is tagged Hip-Hop).
- `popularity` reflects recent plays when the data was collected, so older classics score lower.
- `year` is the album release year, so reissues and compilations can leak old songs.
- Gemini output varies between runs, and it ignores era ("old school rap" doesn't filter years).
- The public demo shares one free-tier Gemini key, so quota can run out. The app then uses the default plan and says so.
- No Spotify playlist export: the app gives track links only.

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/pranav-hariharan/pump-sync
cd pump-sync
uv sync
```

Add your key to a `.env` file (see `.env.example`):

```
GEMINI_API_KEY=your-key
```

Then:

```
uv run streamlit run app.py
```

To rebuild the prepared file from the raw dataset, put `songs.csv` in `data/raw/`, then run `load_all()` and `save_prepared()` from `src/data.py`.

## Project layout

```
app.py             Streamlit app
src/data.py        loading, cleaning, scaling, clustering, prepared file
src/mood.py        Gemini text-to-targets step
src/matching.py    track matching and playlist assembly
src/evaluate.py    baselines and metrics
notebooks/         exploration, clustering, evaluation
```

## Data license

`data/processed/tracks_prepared.csv.gz` is derived from the dataset above and is shared under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for non-commercial use. Changes: filtered by year, popularity and duplicates, columns reduced, cluster labels added.
