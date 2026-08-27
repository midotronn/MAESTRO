# Expert-study music

The live interview editor uses the four full recordings listed in `approved_songs.json`.
The study team confirmed approval to use the full songs on 2026-08-27. Source recordings and
normalized WAV files are intentionally not committed; the manifest records the exact hashes,
durations, and frame targets used on the study pod.

`generate_pop_tracks.py` preserves the deterministic MusicGen fallback used during preparation.
Those temporary tracks are no longer part of the participant-facing catalog.
