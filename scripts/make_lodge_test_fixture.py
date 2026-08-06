"""Regenerate tests/data/lodge_sample_dance.npy -- real LODGE output for the motion-bank tests.

The motion-bank tests otherwise run against MockWindowGenerator, which starts at yaw 0 and dances
on a tidy spot. That is a fine stand-in for splice arithmetic but it cannot show that the bank
survives contact with what the product actually generates, so the suite also carries 24 seconds of
genuine diffusion output: 512 frames turning through 136 degrees and travelling a metre each way.

Run on the pod, which has the FineDance checkpoints and a CUDA torch:

    .venv/bin/python scripts/make_lodge_test_fixture.py

then copy /workspace/e2e/dance.npy to tests/data/lodge_sample_dance.npy. The click track is
synthesised rather than taken from a real song, so the fixture carries no licence encumbrance.
"""
import sys, subprocess, wave
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/AgentLODGE")
sys.path.insert(0, str(ROOT))
WORK = Path("/workspace/e2e"); WORK.mkdir(exist_ok=True)
WAV = WORK / "beat.wav"

SR, DUR, BPM = 22050, 24.0, 112.0
t = np.arange(int(SR * DUR)) / SR
spb = 60.0 / BPM
audio = np.zeros_like(t)
for i in range(int(DUR / spb)):
    s = int(i * spb * SR)
    env = np.exp(-np.linspace(0, 18, int(0.22 * SR)))
    kick = np.sin(2 * np.pi * 55 * np.arange(len(env)) / SR) * env
    audio[s:s + len(kick)] += kick[:max(0, len(audio) - s)][:len(audio) - s] if s + len(kick) > len(audio) else kick
    h = int((i + 0.5) * spb * SR)
    henv = np.exp(-np.linspace(0, 60, int(0.06 * SR)))
    hat = (np.random.default_rng(i).standard_normal(len(henv)) * henv) * 0.25
    if h + len(hat) < len(audio):
        audio[h:h + len(hat)] += hat
chord = sum(0.06 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
audio = np.clip(audio + chord * (0.6 + 0.4 * np.sin(2 * np.pi * t / (4 * spb))), -1, 1)
pcm = (audio / np.abs(audio).max() * 0.9 * 32767).astype(np.int16)
with wave.open(str(WAV), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(f"wrote {WAV} ({DUR}s, {BPM} bpm)", flush=True)

from agentlodge.audio.preprocess import extract_lodge_features
feats = extract_lodge_features(WAV, Path("/workspace/LODGE"))
np.save(WORK / "feats.npy", feats)
print("features", feats.shape, flush=True)

cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_lodge_inference.py"),
       "--agentlodge-root", str(ROOT), "--features-npy", str(WORK / "feats.npy"),
       "--output-npy", str(WORK / "dance.npy"), "--work-dir", str(WORK / "lodge_work"),
       "--lodge-code-path", "/workspace/LODGE",
       "--lodge-weights-path", "/workspace/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
       "--lodge-global-weights-path", "/workspace/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt",
       "--seed", "7"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
print("lodge rc", r.returncode, flush=True)
if r.returncode != 0:
    print("STDOUT:", r.stdout[-3000:]); print("STDERR:", r.stderr[-4000:]); sys.exit(1)
print("dance", np.load(WORK / "dance.npy").shape, flush=True)
print("E2E_GEN_DONE")