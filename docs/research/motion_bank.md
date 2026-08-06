# Named motion bank

MAESTRO includes a manifest-driven bank of 20 common actions for semantic requests such as
"add a clap here", "jump on the next beat", or "insert a wave before the next move". The bank
complements the existing deterministic metric levers and diffusion regeneration:

| Request type | Agent route |
|---|---|
| Change energy, timing, smoothness, or sharpness | Deterministic monotone lever |
| Request a known clap, jump, gesture, step, turn, or level change | Named motion bank |
| Request arbitrary new choreography or variety | LODGE or EDGE regeneration |

## Vocabulary

The initial bank contains single and repeated claps, overhead clap, two jump variants, bounce,
wave, point, hands-up celebration, chest pop, arm punch, side step, step touch, forward and
backward steps, quarter and half turns, body roll, crouch or drop, and rise or reach.

`assets/motion_bank/manifest.json` is the single source of truth. Each entry declares:

- canonical ID, display name, aliases, and category;
- clip path, frame rate, frame count, and semantic event frame;
- mirror and repeat capabilities;
- stationary or traveling behavior;
- source, license, and attribution;
- a declarative semantic validator contract.

The agent exposes one generic `motion_bank` tool. It does not contain one tool or branch per action.
The offline planner resolves aliases from the manifest, and the LLM receives the same vocabulary.

## Timing semantics

Replacement is the default interpretation for ordinary wording such as "add a clap here". The
canonical action is retimed to the selected interval, its semantic event is aligned to a nearby
music beat, and the foreign clip is joined with two-sided seam handling.

Explicit wording such as "insert", "before", "after", or "between" selects fixed-duration
in-window insertion. MAESTRO allocates part of the selected interval to the named action and fits
the original in-window prefix and suffix around it. This preserves:

- the complete dance frame count;
- source audio duration and timing;
- the beat grid and downstream timestamps;
- byte-identical motion outside the selected window.

MAESTRO does not lengthen the global timeline for named-motion insertion because that would
desynchronize the source audio, beat map, cached previews, and comparison renders.

### Closing the window

Whatever the dancer does inside the window, the next window begins where the song left off: the
splice pins the edited window's first and last frames back to the surrounding dance. Anything a
motion still owes at the last frame — a facing, a position, a height — is therefore taken back by
the crossfade over a handful of blend frames. A half turn used to unwind 180 degrees in a quarter
of a second, peaking at 55 rad/s against the song's own 8.9.

`apply()` closes that debt itself, easing the root back over a tail inside the window sized to the
offset and to rates a dancer could actually hold (2.5 rad/s turning, 1.0 m/s travelling, 0.6 m/s
lifting, allowing for a smoothstep peaking at 1.5x its average). A turn is *continued* into a full
revolution whenever that is no further than reversing it, because a dancer finishes a spin rather
than rewinding one — which is why a half turn spliced into a fixed window reads as a full spin and
measures around 2*pi of yaw. The invariant is worth stating plainly: **a motion may travel or turn
inside the window, but it must give the root back before the window ends.**

There is a known edge to this. When the surrounding dance is *itself* turning hard, the closing
rotation and the action's own turn partly cancel, and the action can end up short of its contract:
across 17 windows of real backbone output, `turn_half` in `replace` mode failed validation once, on
the single window where the song turned 89 degrees on its own, measuring 2.567 rad against a 2.6
threshold. `insert` mode, which shifts the surrounding motion instead of pinning it, passed every
window. This is inherent to pinning both edges rather than a defect in the motion, so it is recorded
rather than tuned away — raising the threshold would only hide the tension.

## Validation

Canonical and fitted clips must have shape `(frames, 139)` at 30 FPS using:

`translation(3) | 22 x 6D rotations(132) | contacts(4)`.

The loader rejects missing files, non-finite values, invalid 6D rotations, non-binary contacts,
duplicate aliases, missing provenance, and out-of-range event anchors. Declarative semantic
validators cover:

- upper-body joint activity for claps, waves, points, celebrations, and punches;
- root height and airborne contact intervals for jumps;
- root displacement for locomotion;
- root yaw for turns;
- spine-chain activity for chest pops and body rolls;
- root level direction for drops and rises.

The root contracts measure the **largest excursion from the opening pose**, not the difference
between the first and last frame. An endpoint measure cannot survive the splice: the window's edges
are pinned to the surrounding dance, so a turn or a step that plainly happened reports as zero once
the root closes back. `vertical_peak` already worked this way for jumps, which land where they took
off. Reading a turn out of a spliced window as `yaw change 0.000` is what made the agent tell users
it "couldn't fully satisfy" a spin it had placed perfectly.

The final spliced result is checked again after temporal fitting and seam handling. Unknown or
unsupported actions fail visibly. They never silently select another motion or report success while
keeping the original. Capabilities are the one exception: a request to repeat or mirror an action
whose manifest entry does not allow it is dropped, the action plays once as authored, and the step
note says so — failing the whole edit would hand back an unchanged window, which serves the user
worse than the nearest valid edit. The planner is told which motions accept `repeats` and `mirror`
so it can choose a better one, but the degradation is what makes it safe when it does not.

Three posture invariants are enforced for every canonical clip, independently of the per-action
validators:

- planted feet rest on the floor. Root height is solved from the pose by `_ground` rather than
  authored by hand, because a hand-picked drop distance sinks the feet through the ground whenever
  it disagrees with how much the bent legs actually shorten;
- at least ten joints move meaningfully, so no action reads as a mannequin with one moving limb;
- lateral steps open the stance instead of leaning both legs the same way.

The manifest validators are generic shape checks, so they constrain magnitude rather than meaning:
`joint_activity` asks only that the named joints move, which a clap satisfies whether or not the
hands ever meet, and a side point satisfies while sweeping diagonally forward. `root_displacement`
and `root_yaw` are **unsigned excursions**, so `step_forward` and `step_backward` — which share an
axis and a threshold — validate each other's clips exactly. Where an action's name makes a specific
claim, that claim is pinned by a test instead — claps close at the event frame and land in front of
the chest, the side point stays lateral, jumps are airborne at their accent, a bounce travels far
enough to see, no raised hand is stranded behind the back, and each named step travels the way its
name says both as a bank clip and after it has been spliced into a song — because the event frame is
what the editor snaps to a beat and is therefore the frame the viewer actually reads. **Validators
pin magnitude; tests pin meaning.**

Those tests ran only against `MockWindowGenerator` for most of the bank's life, which is a tidy
base: it starts at yaw 0 and dances on the spot. `tests/data/lodge_sample_dance.npy` is 24 seconds
of real LODGE diffusion output — 512 frames from a 112 bpm click track, turning through 136 degrees
and travelling a metre each way — so the bank is also exercised against what the product actually
generates. Regenerate it with `scripts/make_lodge_test_fixture.py`, which synthesises the click track,
extracts librosa features and runs `run_lodge_inference.py` against the FineDance checkpoints on
the pod. The track is synthesised rather than sampled so the fixture carries no licence.

## Frame conventions

Three separate coordinate conventions meet in `scripts/build_motion_bank.py`, and they do not agree
with one another, plus two more that bite outside it. Getting one wrong produces clips that pass
every validator while doing the opposite of what they are named, so each is stated once and derived
from the skeleton rather than remembered:

- the **SMPL joint template faces native +Z**, with +X to the dancer's left and +Y up. IK targets
  passed to `_solve_arm` are absolute positions in that frame, so **+z is in front of the dancer**;
- `to_zup` maps `y_zup = -z_native`, so in the stored Z-up editing frame **+y is *behind* the
  dancer**. Travelling forwards means *subtracting* from `trans[:, 1]`, which is why root travel
  goes through the `_travel()` helper instead of being written inline;
- spine and hip pitch (`aa[:, j, 0]`) is **positive = leaning forwards**, which is the opposite sign
  to the root translation it usually accompanies.

An early version of the bank was authored believing native -Z was forward. Every clap, punch and
reach happened behind the body, and `step_forward` moonwalked. Facing is now measured off the
skeleton in tests (`_body_forward` derives it from the ankle-to-toe vector) rather than assumed.

A fourth trap is vertical: `_ground` re-solves root height from the leg pose on every grounded
frame, so an authored `trans[:, 2]` rise is silently cancelled whenever the feet stay planted. A
bounce or a dip has to come from knee flexion, not from the root.

A fifth is that the *validator* has a frame too, and it is not the same one the clip is in by the
time it runs. `insert` yaw-aligns the action to whatever direction the dancer faces at the splice
point, so a `root_displacement` contract read along a fixed world axis measures the travel only
while the dancer happens to face down that axis. Every travelling motion failed validation at 45,
90, 225 and 270 degrees of song rotation and passed at 0 and 180 — a user-visible edit that worked
or failed on nothing they could see. `root_displacement` is now projected onto the dancer's own
lateral and sagittal directions, derived from the clip's opening root yaw, so the check means the
same thing wherever the dancer is pointing. A world axis is never the right frame for a claim about
a body.

## Authoring style

Clips are authored from a shared "ready" stance — arms carried below horizontal with soft elbows and
slightly bent knees — rather than the SMPL T-pose, and every clip carries a small endpoint-neutral
performance layer (breathing, weight shift, torso counter-rotation). Accents wind up before the hit
and settle after it, and locomotion carries opposite-arm swing. Because the performance layer
vanishes at the first and last frame, validators that compare clip endpoints measure exactly what
the authoring recipe produced.

## Authoring and licensing

Run `python scripts/build_motion_bank.py` to regenerate every canonical `.npy` clip and validate the
whole manifest. The initial clips are procedurally authored for MAESTRO and distributed under MIT.

New clips may be self-authored or imported from a source that explicitly permits redistribution.
Record the exact source, license, attribution, and any modifications in the manifest. Do not commit
AMASS, BABEL, FineDance, HumanML3D-derived arrays, SMPL assets, or a raw Mixamo motion pack without
separate permission and licensing review. Clearly licensed AIST++ excerpts may be added with
CC BY 4.0 attribution after their action identity and output quality are reviewed.

## Adding a motion

1. Add a unique manifest entry and avoid aliases owned by another action.
2. Add a deterministic authoring or import recipe.
3. Generate the canonical clip with `scripts/build_motion_bank.py`.
4. Choose a validator type and threshold that the canonical clip passes for the intended reason.
5. Add the action to the table-driven planner, fitting, insertion, and adversarial tests by relying
   on manifest enumeration rather than writing a new executor branch.
6. Render and visually review the action before publishing it. `scripts/build_motion_bank_reel.py`
   stitches the whole bank into one continuous take with the editor's own seam blend, and
   `scripts/render_motion_bank_reel.sh` renders it on the GPU pod as the Y-Bot with each action's
   name burned in, so a new action is judged against the rest of the vocabulary in one pass rather
   than as an isolated clip. Numeric checks are the gate, not the review: a 2D contact sheet
   flattens depth badly enough to invent problems that measurement disproves.
