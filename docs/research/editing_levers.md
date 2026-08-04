# Editing levers: satisfying NL dance edits without infinite generation

## Question

Can a natural-language dance edit ("more energetic", "calmer", "smoother", "snappier",
"tighter to the beat") be **guaranteed** to satisfy the user's request by the end of the
agent loop, as fast as computationally possible, WITHOUT falling back to sampling the
diffusion backbone over and over until a metric happens to move?

## Why pure regeneration is not enough

The diffusion backbones (LODGE, EDGE) are **music-conditioned**. The `energy` guidance
knob is honored only by the mock generator; the real backbones sample motion whose
intensity is bounded by what the music section supports. So "more energetic" on a calm
section can regenerate forever and honestly hold the same energy. Regeneration is the
right tool for **variety** (genuinely new movement) but it cannot **guarantee** a metric
target, and best-of-K only searches, it does not control.

## The key observation: the metrics are levers

The window metrics are simple functionals of the pose sequence:

- `energy = mean || pose[t] - pose[t-1] ||`  (mean per-frame speed)
- `jerk   = mean || third difference of pose ||`
- `bas`    = beat-alignment score (accent frames vs. music beats)
- `foot`   = foot-contact consistency

Because energy is literally the mean per-frame speed, it can be moved **monotonically** by
scaling the pose's deviation from a smooth baseline. This is the classic
**motion-signal-processing** result: decompose a motion into a low-frequency band (posture)
plus higher-frequency bands (the dance dynamics) and scale the bands independently.

## Literature basis

- **Bruderlin and Williams, 1995, "Motion Signal Processing".** Multiresolution filtering
  and multitarget interpolation: amplify or attenuate frequency bands of a motion to change
  its expressive intensity. This is exactly a per-band gain on the deviation from a low-pass
  baseline.
- **Witkin and Popovic, 1995, "Motion Warping".** Smooth, keyframe-anchored deformation of a
  motion signal that preserves the fine structure. Our beat-align time-warp is a monotone,
  anchor-based reparameterization in this family.
- **Unuma, Anjyo, Takeuchi, 1995, "Fourier Principles for Emotion-based Human Figure
  Animation".** Emotional/effort content lives in specific frequency bands; scaling them
  changes "tired" vs. "energetic" without changing the choreography.
- **Laban Movement Analysis / Effort (Weight, Time, Space, Flow).** "Energetic" maps to
  strong Weight and sudden Time, i.e. larger, faster deviations; "calm" is light and
  sustained. The energy lever is a computational Effort-Weight/Time control.
- **Diffusion editing (MDM, EDGE editing, guided sampling, DNO).** Guidance/inpainting can
  steer generation toward attributes, but it is (a) slower and (b) still bounded by the
  conditioning. We keep regeneration for **content/variety**, not as the metric controller.

Conclusion from the review: for the metrics we grade on, a small set of **deterministic,
monotone primitives** exists and is well founded. Infinite generation is unnecessary for
metric-directed requests; it is reserved for "give me different moves".

## The lever set (one monotone lever per metric)

| Request | Lever | Metric moved | Guarantee |
|---|---|---|---|
| tighter / on the beat | `beat_align` (PCHIP monotone time-warp onto beats) | bas up | monotone in passes/strength |
| smoother / flowing | `smooth` (geodesic low-pass) | jerk down | monotone in amount |
| more energetic / bigger | `energy` up (`accentuate` gain > 1) | energy up | monotone in gain |
| calmer / softer | `energy` down (`accentuate` gain < 1) | energy down | monotone in gain |
| snappier / punchier | `sharpen` (`accentuate`, narrow baseline) | jerk up | monotone in gain |
| different / freestyle | `regenerate` (diffusion, warm) | none (variety) | any fresh sample succeeds |
| reverse / mirror | `retrograde` / `mirror` | none (exact) | deterministic |

### `accentuate`: the correct energy lever

The retired `amplitude_scale` scaled every pose about **one global mean pose**. That drags
the whole body toward an average posture, is **off-manifold** for rotations, and injects
jerk. It looked like a jittery or sped-up copy, so energy was made generation-only.

`accentuate` (agentlodge/dance/transition.py) fixes this:

1. Decompose the window into a **geodesic low-pass baseline** (slow postural trajectory) plus
   a residual (the actual dance dynamics).
2. Scale ONLY the residual, **per frame**: rotations via the geodesic residual
   `q_base^{-1} q` (stays on SO(3)); translation via the Euclidean residual.
3. **Taper** the gain to 1.0 over a few frames at both ends so the spliced seams are
   byte-identical to the original (verified: outside-window diff = 0 at every gain).
4. Optionally damp root-translation scaling to limit foot sliding.

Because the residual carries the per-frame speed, `energy` moves monotonically with the
gain. Measured on a real Treasure window: gain 0.55 to 1.9 gives energy 0.29 to 0.63,
strictly increasing, feet stable, seams intact. It does NOT time-warp, so there is no
sped-up-copy look, and it is on-manifold, so there is no jitter (the smoothing finisher and
jerk guard bound any residual jerk).

## The loop: guaranteed convergence

For every metric-directed request the agent now:

1. Plans the dedicated monotone lever (LLM reasoning, or the offline keyword planner).
2. Executes it; the executor rejects a step that moves its OWN target the wrong way (cannot
   happen for a monotone lever, but keeps the contract honest).
3. Verifies EVERY requested metric on the spliced window.
4. On a miss, escalates the lever's scalar (larger gain / more passes) and retries; because
   the lever is monotone, escalation always advances the metric.
5. A smoothing finisher pulls jerk back toward baseline WITHOUT undoing the primary goal, and
   an artifact guard forbids shipping a jittery or foot-skating window.

Result: metric-directed edits are satisfied on the first attempt, instantly (no GPU), on any
window regardless of the music. Verified battery on the real Treasure dance (offline, no
backbone): more energetic, much more energetic, calmer, smoother, snappier, tighter to the
beat, and "energetic and on beat" all satisfied (ok, not kept-original). Variety requests use
the warm regeneration daemons (about 2 to 4 seconds). This is as fast as computationally
possible: metric tweaks are instant, and only genuinely new choreography pays the diffusion
cost.
