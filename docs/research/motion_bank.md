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

The final spliced result is checked again after temporal fitting and seam handling. Unknown or
unsupported actions fail visibly. They never silently select another motion or report success while
keeping the original.

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
6. Render and visually review the action before publishing it.
