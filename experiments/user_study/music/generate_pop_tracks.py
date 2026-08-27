"""Generate original high-energy pop instrumentals for the expert-study editor."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


MODEL = "facebook/musicgen-medium"
TOKENS_PER_SEGMENT = 1205
CROSSFADE_SECONDS = 2.0
TRACKS = {
    "golden-pulse": {
        "name": "Golden Pulse",
        "prompt": (
            "original high-energy funk-pop instrumental, 118 BPM, punchy live drums, "
            "syncopated electric bass, bright brass stabs, clean rhythm guitar, celebratory "
            "arena-sized chorus, polished modern production, no vocals"
        ),
        "seeds": [1701, 1702, 1703],
    },
    "neon-weekend": {
        "name": "Neon Weekend",
        "prompt": (
            "original upbeat disco-pop instrumental, 124 BPM, four-on-the-floor drums, glossy "
            "synth chords, funky bass guitar, handclaps, short horn accents, energetic hook, "
            "polished dance-floor production, no vocals"
        ),
        "seeds": [2401, 2402, 2403],
    },
    "city-heat": {
        "name": "City Heat",
        "prompt": (
            "original modern funk-pop instrumental, 122 BPM, tight acoustic drums, slap bass, "
            "muted rhythm guitar, bold brass section, rhythmic breaks, uplifting chorus, "
            "high-energy polished production, no vocals"
        ),
        "seeds": [3301, 3302, 3303],
    },
    "electric-bloom": {
        "name": "Electric Bloom",
        "prompt": (
            "original high-energy dance-pop instrumental, 128 BPM, driving kick and snare, "
            "sparkling synthesizers, disco bass line, handclaps, brass flourishes, huge upbeat "
            "chorus, polished radio production, no vocals"
        ),
        "seeds": [4201, 4202, 4203],
    },
}


def crossfade(parts: list[np.ndarray], sample_rate: int) -> np.ndarray:
    overlap = int(round(CROSSFADE_SECONDS * sample_rate))
    merged = parts[0].astype(np.float32, copy=True)
    phase = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    fade_out = np.cos(phase)
    fade_in = np.sin(phase)
    for part in parts[1:]:
        part = part.astype(np.float32, copy=False)
        if len(merged) < overlap or len(part) < overlap:
            raise ValueError("generated segment is shorter than the requested crossfade")
        blend = merged[-overlap:] * fade_out + part[:overlap] * fade_in
        merged = np.concatenate([merged[:-overlap], blend, part[overlap:]])
    return merged


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples * (0.95 / peak)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def generate(track_id: str, output_dir: Path) -> dict:
    spec = TRACKS[track_id]
    if not torch.cuda.is_available():
        raise RuntimeError("MusicGen requires the CUDA-enabled PyTorch installed by pod setup")
    processor = AutoProcessor.from_pretrained(MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    inputs = processor(
        text=[spec["prompt"]],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    segments = []
    with torch.inference_mode():
        for seed in spec["seeds"]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            audio = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                top_k=250,
                temperature=1.0,
                max_new_tokens=TOKENS_PER_SEGMENT,
            )
            segments.append(audio[0, 0].float().cpu().numpy())

    sample_rate = int(model.config.audio_encoder.sampling_rate)
    samples = crossfade(segments, sample_rate)
    path = output_dir / f"{track_id}-raw.wav"
    write_wav(path, samples, sample_rate)
    result = {
        "id": track_id,
        "name": spec["name"],
        "model": MODEL,
        "model_license": "CC-BY-NC-4.0",
        "prompt": spec["prompt"],
        "seeds": spec["seeds"],
        "sample_rate": sample_rate,
        "duration_seconds": round(len(samples) / sample_rate, 3),
        "raw_wav": str(path),
    }
    (output_dir / f"{track_id}.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=sorted(TRACKS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.track, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
