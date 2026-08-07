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
- a declared musical duration (`recommended_beats`);
- composition ownership: absolute and additive joints, translation axes, contacts, and whether a
  turn carries its root heading into the suffix;
- optional event-pose joints that take the shortest path to the beat pose instead of replaying a
  long authored wind-up after beat compression;
- source, license, and attribution;
- a declarative semantic validator contract.

The agent exposes one generic `motion_bank` tool. It does not contain one tool or branch per action.
The offline planner resolves aliases from the manifest, and the LLM receives the same vocabulary.

## Timing semantics

Replacement is the default interpretation for ordinary wording such as "add a clap here". The
action plays for its declared `recommended_beats`, and only the joints and root channels listed in
its composition metadata are layered onto the host — everything else stays the song's own
choreography. The authored clip's frame count defines the *shape* of the action; its recommended
beats define playback duration. Its semantic event is aligned to a nearby music beat.

The action's duration is deliberately not a function of the selection. Retiming the clip to fill
whatever interval the user happened to drag makes its speed an accident of the gesture that
selected it: a 1.5 s clap dropped on a 4 s selection played at 0.4x, smeared across nearly eight
beats, and read as slow motion fighting the song rather than as dancing with it. At a 15-frame
beat, a one-beat clap is therefore 15 frames even though its canonical source contains 45 frames.
This is intentional retiming to the music contract, not truncation.

Retiming every authored in-between pose is still wrong for a one-beat pose accent. The source clips
begin from a shared ready stance, so compressing a full overhead-clap, punch, or arm-raise wind-up
into 15 frames makes the host visit poses that belong to the source actor, not to the dance being
edited. `event_pose_joints` identifies joints whose only contract is the pose on the beat. They
travel directly from the host pose to that event pose and back under the composition envelope.
Joints with meaningful internal timing retain the clip trajectory: the hand oscillation of a wave,
the three contacts of a repeated clap, and the root and leg phases of jumps and level changes.

Beat-locking never fills the window to its last frame. A few frames of real dance are always kept
after the action for the hand-back to fade into; when an action finished two frames from the edge
the whole pose difference had to be closed inside those two frames and the dancer was flung at
nearly six times any speed in the song. Dropping a beat from the gesture is far cheaper than that.
The bank receives the complete beat grid re-expressed relative to the selection, not only beats
inside it. A short selection with one visible beat therefore still knows the song's period, while
a selection with no feasible beat reports a nonzero event error instead of claiming alignment.
An absent or empty beat grid means the song has no usable timing map and does not fail an otherwise
valid action; only a nonempty grid with no reachable beat is an alignment failure.

Explicit wording such as "insert", "before", "after", or "between" selects fixed-duration
in-window insertion. MAESTRO allocates part of the selected interval to the named action and fits
the original in-window prefix and suffix around it. This preserves:

- the complete dance frame count;
- source audio duration and timing;
- the beat grid and downstream timestamps;
- byte-identical motion outside the selected window.

MAESTRO does not lengthen the global timeline for named-motion insertion because that would
desynchronize the source audio, beat map, cached previews, and comparison renders.

### Channel ownership

Canonical clips all begin from one generic ready stance. Replacing the whole skeleton with that
clip made a valid clap look wrong in the editor: for roughly three beats it erased the host's
footwork, root path, contacts, torso rhythm, and style before returning to the song. The hands did
meet, but the dancer abandoned the choreography to do it. On the exact rejected TRS edit, non-arm
velocity correlation fell to `0.09`, the root moved `0.169 m`, and 152 contact bits changed.

Composition metadata makes the ownership explicit:

- arm gestures own only their relevant collar, shoulder, elbow, and wrist chains, with limited
  additive torso response where needed;
- jumps own the whole-body jump delta, vertical translation, and contacts;
- steps own the leg chains, their declared travel axis, contacts, and selected counter-motion;
- turns own root yaw, leg chains, selected counter-motion, and carry the heading into the suffix;
- undeclared joints remain byte-identical, contacts change only when declared, and root
  translation changes only on declared axes.

The bank composes once. `run_agent_edit` must not treat the result as foreign generated motion and
send it through `splice_window` again; that second splice was changing the already-composed root
and contacts a second time. Temporal transforms run before the bank so its action range and event
metadata describe the final window. The remaining outer boundary crossfade is capped separately on
each side at the action range: it may blend the host into the surrounding song, but it may never
overlap and dilute the semantic pose. This matters on a 30-frame selection, where a symmetric
15-frame crossfade otherwise covers every frame and can open a valid clap back to `0.47 m`.

### Closing the window

Whatever the dancer does inside the window, the next window begins where the song left off: the
splice pins the edited window's first and last frames back to the surrounding dance. Anything a
motion still owes at the last frame — a facing, a position, a height — is therefore taken back by
the crossfade over a handful of blend frames. A half turn used to unwind 180 degrees in a quarter
of a second, peaking at 55 rad/s against the song's own 8.9.

`apply()` closes that debt itself, easing the root back *after the action* over a tail inside the
window sized to the
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

### Handing over at a seam

Splicing at the action's own length puts two seams *inside* every window where filling the window
put none. Both are handed over by fading the pose difference out while the incoming motion carries
on, rather than by easing toward the outgoing pose.

The distinction matters more than it sounds. Landing the incoming clip *on* `left[-1]` makes the
first joined frame a pose-for-pose copy of the last one, so a frame of time passes with nobody
moving and the dance then rushes to catch up: measured joint speed was exactly 0.000 at every
seam, followed by a spike to twice the song's own peak. A held frame reads as a dropped frame or a
hitch, which is far more visible than a fast one. The join therefore targets where the outgoing
motion was *going* — `left[-1]` advanced by one frame of its own velocity — so the seam frame
advances like any other.

The hand-over is as long as the pose gap needs, bounded at one beat. A fixed length closes a small
gap gently and a large one violently, but an unbounded one would fade a clip's offset out across
seconds of the song's own choreography. With both in place the worst seam across all twenty
motions sits at 1.4x the song's own 99th-percentile joint speed, against 3.0x before.

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

The final spliced result is checked again after temporal fitting and seam handling. A beat-anchored
named action is graded by its event-to-nearest-beat error (at most half a frame), not by whole-window
BAS: one correctly timed clap can change unrelated velocity peaks and lower an aggregate score even
though the clap itself landed exactly on the beat. Unknown or
unsupported actions fail visibly. They never silently select another motion or report success while
keeping the original. Capabilities are the one exception: a request to repeat or mirror an action
whose manifest entry does not allow it is dropped, the action plays once as authored, and the step
note says so — failing the whole edit would hand back an unchanged window, which serves the user
worse than the nearest valid edit. The planner is told which motions accept `repeats` and `mirror`
so it can choose a better one, but the degradation is what makes it safe when it does not.

Each manifest entry also declares a `default_anchor`. Discrete hits and arrivals, including claps,
jumps, points, celebrations, chest pops, punches, drops, and rises, default to `beat`; their semantic
event is placed on the nearest feasible musical beat. Grooves and multi-beat phrases default to
`center`. This is enforced after either planner runs, so an LLM cannot move a clap off beat merely by
returning an arbitrary `center` anchor. An explicit user placement such as before, after, start,
center, or end still overrides the motion default.

That BAS exception applies only when "on the beat" anchors the named event. An explicit compound
request such as "add a clap, then make the rest of the window more on beat" retains its whole-window
BAS goal and is verified separately. Likewise, composed `joint_activity` and
`articulation_chain` validators consider rotation relative to the host as well as absolute rotation.
An arbitrary host pose can cancel an absolute joint angle even when the final FK geometry clearly
depicts the requested point or punch; validation must not reject the visible action for that.

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

A proximity claim needs a floor as well as a ceiling. The clap test asserted only that the hands
finished closer than 0.12 m, and two hands driven to a *single point* satisfy that better than two
hands meeting: `_clap_arms` gave the left and right IK solvers the same target, so every clap in
the bank rendered as crossed, interpenetrating forearms with a measured gap of 0.0001 m while both
the validator and the test passed. Clap targets are now mirrored either side of the midline like
every other two-armed recipe, and the gap has both a positive floor and a ceiling.

Even that is not a complete hand-contact contract. The exact live Treasure edit later measured a
valid `0.094 m` wrist gap while the Y-Bot showed one palm turned up and the other sideways. SMPL
joint FK ends at the wrists, so every position-only test and skeleton sheet missed the visible
misalignment. Clap authoring now drives the wrist rotations as well as the shoulder/elbow IK:
palms face one another and the fingers run parallel at the event. Regression tests compose the
global wrist frames and assert both directions across canonical clips, splice modes, and eight host
headings; the final check remains a side-view Y-Bot render because only the skinned hands show the
actual contact surfaces.

The same review now covers every motion that can affect a hand. Ten motions do not own either
wrist channel (`bounce_in_place`, `chest_pop`, `side_step`, `step_touch`, `step_forward`,
`step_backward`, `turn_quarter`, `turn_half`, `body_roll`, and `crouch_drop`), so composition tests
require their wrist data to remain host-identical. The other ten have explicit contracts:

- claps require opposing palm normals and parallel fingers at contact;
- `jump_two_foot` inherits the host wrist pose rather than introducing a new one;
- `wave`, `point_side`, `arm_punch`, `jump_arms_up`, `celebrate_hands_up`, and `rise_reach`
  keep the rigid open hand anatomically continuous with the forearm, with wave as the only
  deliberate local wrist oscillation;
- all ten are checked for abrupt local wrist steps on several phases of real LODGE output in both
  replacement and insertion modes, while authored hand planes are checked across eight headings
  and mirrored variants.

These tests intentionally measure local wrist rotation. A global wrist-frame delta also contains
the shoulder and elbow motion that carried the hand through space, so treating it as wrist bend
produces false failures on reaches, celebrations, and arm raises. Close front/side Y-Bot temporal
sheets remain the visual authority for palm shape and continuity because the 22-joint skeleton
still ends at the wrist.

**Hands meeting is not enough; *where* and *how* they meet is the action.** With the gap fixed,
every clap still passed every check while rendering as a bow: the palms met above the chest at
collarbone height with the elbows winged out level with the shoulders. The user rejected it on
sight as "not a clap at all". Three separate authoring errors produced it, none visible to any
magnitude check:

- **Height.** The target was `y=+0.04` in the template frame. The shoulders sit at `+0.083` and
  the chest at `-0.057`, so it landed *above* the chest, under the chin. A clap belongs a little
  below the chest line — `_CLAP_POINT`.
- **Elbow swivel.** Two arm poses reach the same hand target, and `_solve_arm` chose between them
  with one hint vector that pointed straight out sideways for every motion. For a hand carried in
  front of the sternum that lifts the elbows to shoulder height and splays them wide. The hint is
  now a parameter (`_ELBOW_OUT` default, `_CLAP_ELBOW` for claps).
- **Between the claps.** `clap_repeat` drove the arms straight off its hit pulses, so they
  returned to the rest stance after every clap and swung the full 0.84 m of stance width three
  times over — flapping, not clapping. `_CLAP_GUARD` holds them up across the phrase, parting
  about 0.27 m per clap, while the envelope still vanishes at both ends so the clip starts and
  finishes on the stance the splice hands over on.

Judge posture in the **dancer's own frame**, never world Z. The host dance leans and twists the
torso, so reading "hands above the shoulders" off world vertical charges that lean to the clip.
The relevant upright for an inherited arm pose is the upper-torso frame from chest to shoulder
girdle, not a pelvis-to-neck axis that can point diagonally across a crouched host. The same
bit-identical spliced arm pose otherwise measured as though it changed height from window to
window. This is the same error as grading a spliced clip's speed against the song's own.

The generic version of this trap: an elbow at shoulder height is correct for an *extended* arm
(a punch, a side point, a reach) and wrong only for a *folded* one. A check that ignores the
distinction flags eight of the twenty motions and buries the one real defect in false positives.

**The clap was not the only one.** Reviewing all twenty the same way turned up a second motion
doing the wrong action while passing everything: `chest_pop` never moved the chest. At its event
frame spine3 sat at `+0.018 m` — its rest offset, unchanged to the millimetre — while the head
speared `0.272 m` forward and dropped `0.134 m`. It read as a pigeon peck. The cause is a
kinematic fact worth stating plainly:

> **Only joints *below* a body part can move that part.** Rotating spine3 or the neck swings the
> head; it cannot translate the chest. A chest pop has to be driven from the pelvis, spine1 and
> spine2, with spine3 and the neck giving that rotation back so the head arrives level.

Cancelling the rotation exactly is still not enough — the chain below has already carried the head
forward, so the counter has to *over*-rotate to hold it in place. The fixed clip travels `0.066 m`
at the chest against `0.021 m` at the head; the shipped one was `0.001 m` against `0.108 m`.

Neither the `articulation_chain` validator nor `test_every_motion_animates_a_meaningful_share_of_the_body`
could ever have caught this: both only ask whether the listed joints moved, and a nod moves them
just as convincingly as a pop. **Every motion needs one assertion naming the thing its own name
promises** — the chest leads for `chest_pop`, the wave travels up the spine in order for
`body_roll`, the hands meet in front of the chest for a clap. Magnitude checks are necessary and
never sufficient.

Nothing in that failure was subtle at full size — it survived because the twenty-per-page contact
sheet used to review the bank renders each action too small to see a hand at.
`scripts/review_motion_visually.py` draws one motion per sheet from four angles, forces the
closest-approach frame into the sample rather than sampling evenly past it, and keeps world height
so a jump still looks like a jump. Its skeleton still stops at the wrist, however, so hand-contact
motions also require a close front/side Y-Bot render. Review new clips with both, not with the reel
alone.

Those tests ran only against `MockWindowGenerator` for most of the bank's life, which is a tidy
base: it starts at yaw 0 and dances on the spot. `tests/data/lodge_sample_dance.npy` is 24 seconds
of real LODGE diffusion output — 512 frames from a 112 bpm click track, turning through 136 degrees
and travelling a metre each way — so the bank is also exercised against what the product actually
generates. Regenerate it with `scripts/make_lodge_test_fixture.py`, which synthesises the click track,
extracts librosa features and runs `run_lodge_inference.py` against the FineDance checkpoints on
the pod. The track is synthesised rather than sampled so the fixture carries no licence.

The fixture is raw LODGE data: `[contacts(4) | translation(3) | rotations(132)]`, Y-up. The editor
uses `[translation(3) | rotations(132) | contacts(4)]`, Z-up. The renderer auto-detected both,
which masked the boundary bug; the editor did not, so it interpreted contacts as root translation
and shifted every rotation channel. `to_editor139` is now the canonical boundary for persisted
generation output, session loading, direct editor calls, and tests.

For visual review of the composed path, `scripts/scratch/review_composed_bank.py` produces one
before/after sheet per action on the exact live TRS dance, while
`scripts/scratch/render_composed_bank_video.py` renders all twenty at real 30 FPS in front and side
views. Static sheets establish posture and channel ownership; the playback reel establishes timing
and continuity. Both use one camera anchored at the action start rather than recentering every
frame, so jumps, drops, and locomotion remain visible instead of being erased by the audit view.

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

A fifth is that the *validator* has a frame too, and it is not the same one the clip is in by thetime it runs. `insert` yaw-aligns the action to whatever direction the dancer faces at the splice
point, so a `root_displacement` contract read along a fixed world axis measures the travel only
while the dancer happens to face down that axis. Every travelling motion failed validation at 45,
90, 225 and 270 degrees of song rotation and passed at 0 and 180 — a user-visible edit that worked
or failed on nothing they could see. `root_displacement` is now projected onto the dancer's own
lateral and sagittal directions, derived from the clip's opening root yaw, so the check means the
same thing wherever the dancer is pointing. A world axis is never the right frame for a claim about
a body.

A sixth bites when *measuring* rather than authoring, and it fails silently. `compute_poses`
returns joints in the Z-up editing frame, so the dancer's forward is `-y` and `+z` is up. Building
a body frame the usual way — `fwd = normalize(axis - up * (axis @ up))` — with `axis` mistakenly
set to `+z` projects it onto the up vector and leaves *zero*, which normalizes into pure floating
point noise. It does not raise; it yields a unit vector pointing somewhere arbitrary, and every
"forward" number computed from it comes out as a small, stable, entirely plausible float. Derive
forward from the skeleton, or check it against `step_forward` (pelvis travels `y=-0.460`) and
`step_backward` (`y=+0.460`), before trusting a single measurement taken in that frame.

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
