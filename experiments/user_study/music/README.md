# Expert-study music

The study asset registry retains the four approved recordings listed in `approved_songs.json`.
The participant-facing editor exposes only the five shorter Dynamite sections; the other
recordings remain available to the frozen comparison workflow but are hidden from the editor.
The study team confirmed approval to use the recordings on 2026-08-27. Source recordings and
normalized WAV files are intentionally not committed; the manifest records the exact hashes,
durations, and frame targets used on the study pod.

`generate_pop_tracks.py` preserves the deterministic MusicGen fallback used during preparation.
Those temporary tracks are no longer part of the participant-facing catalog.
