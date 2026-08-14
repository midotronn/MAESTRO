# Strengthening Choreography in AgentLODGE, Research & Design

> Status: **research/design** (branch `choreography-research`). This document synthesizes a
> literature review across five threads and proposes a grounded, phased design. It is a plan,
> not an implementation. Full per-thread literature reviews with citations are preserved
> alongside this work (see *References* and the session `lit_reviews/` artifacts).

## 0. Motivation & current state

AgentLODGE already turns a song into a *structured* dance: `structure.py` detects musical
sections + an energy arc, `storyboard.py` has an LLM author a per-section plan, and `story.py`
assembles LODGE/EDGE material per section with inertialized seams. This document proposes five
enhancements to make the choreography meaningfully stronger.

**What already exists (reuse it):**
- Musical-form analysis: `agentlodge/audio/structure.py` (agglomerative segmentation over
  chroma+MFCC, energy arc = RMS+flux, repetition labels via feature cosine-sim, roles, climax).
- LLM plan: `agentlodge/agent/storyboard.py` (arc + per-section role/intensity/vocabulary/
  generator_bias/reuse/variation).
- Assembly + primitives: `agentlodge/dance/story.py` (cost-based per-section selection +
  inertialized blend), `transition.py`, **`mirror` (spatial L/R), `retime`, `amplitude_scale`,
  `blend_onto`, `rotate_root_yaw` already implemented**; motif reuse is already wired through the
  storyboard `variation` dict.
- Structure metrics: `agentlodge/dance/story_metrics.py` (arc adherence, sectional contrast,
  motif recurrence, boundary alignment, seam jerk).

**Key gaps this design targets:**
1. Structure detection is local/energy-driven; it under-uses **sectional repetition** (which
   sections are the *same*) and never exploits it for **recapitulation/mirroring** (ABA).
2. **No beat-alignment metric exists** anywhere in the codebase, the exact weakness the user
   flagged is not even measured.
3. The LLM reasons over **numbers only**; candidate segments are never *described*.
4. Diffusion generation uses **one seed**, the stochasticity of LODGE/EDGE is untapped.
5. There is **no interactive editing** path, the user cannot refine a result in natural language.

---

## 1. What makes a song "structured" (Thread 1)

### Findings
- **Musical form is hierarchical**: beat → measure → hypermeasure (4-bar) → phrase (8–16 bar) →
  section → whole-song form (ABA/rondo/AABA/EDM-drop). Lerdahl & Jackendoff's *GTTM* (1983)
  formalizes grouping + metrical structure + tension/release; Huron's *Sweet Anticipation* (2006)
  shows **repetition creates expectation and the return of an earlier section yields
  satisfaction**, i.e., recapitulation is perceptually the strongest structural device.
- **MSA algorithms**: Foote-novelty over a self-similarity matrix (SSM) for boundaries;
  **McFee & Ellis (ISMIR 2014) Laplacian spectral clustering** assigns *section-type identity*
  (A/B/A/C…), not just boundaries, so it detects that "chorus @30s == chorus @90s". Toolkits:
  `librosa.segment.recurrence_matrix`/spectral clustering, `MSAF` (`boundaries_id="sf"`,
  `labels_id="scluster"`). **Jukebox embeddings** (already used by EDGE) beat MFCC/chroma for
  structural feeling.
- **Hypermeter matters**: sections/drops begin on 4/8-bar downbeats; boundary snapping should
  respect hypermetric phrasing, not just any downbeat.

### Design
Deepen `structure.py` from "energy-segmentation" to "form-aware":
- **S1. Section-type labels via spectral clustering.** Replace/augment the current greedy
  cosine-sim `_label_sections` with a Laplacian/spectral-clustering labeler (McFee & Ellis) so
  same-type sections get the same label robustly. Optionally build the SSM from **Jukebox
  features** (already extracted for EDGE) instead of chroma+MFCC.
- **S2. Hypermeter-aware boundaries.** Snap boundaries to 4/8-bar hypermetric downbeats (derive a
  downbeat grid from tempo/beat tracking) rather than arbitrary downbeats.
- **S3. Expose a `repeat_of` field** per `Section` (earliest same-label section) directly in
  `MusicStructure`, so the storyboard and assembler can *act* on repetition (feeds Thread 2).
- **S4. Keep the robust fallback** (downbeat/uniform) unchanged.

---

## 2. Motif recurrence, recapitulation & mirroring, ABA (Thread 2)

### Findings
- Dance-composition scholarship (Blom & Chaplin *The Intimate Act of Choreography*; Humphrey
  *The Art of Making Dances*; Smith-Autard; Laban) names the exact devices the user asked for:
  **ABA / recapitulation** (restate A after B), **retrograde** (temporal reversal, "mirror in
  time"), **spatial mirroring** (L/R reflection), **augmentation/diminution** (slower/faster =
  retime), **inversion**, **fragmentation**.
- Closest prior systems, Aristidou et al. *"Rhythm is a Dancer"* (TVCG 2022, pose→motif→
  choreography hierarchy) and *"Music-to-Dance via Atomic Movements"* (2025, LLM-plan → synthesis)
 , validate the **plan-then-realize** paradigm AgentLODGE already uses, **but none implement
  explicit ABA recapitulation or retrograde/mirror as principled variation.** That is our novelty.
- The pipeline **already has `mirror` (spatial), `retime`, `amplitude_scale`** and a reuse path in
  `select_sources` (`reuse:{i}` with mirror/retime/amplitude). **Missing: temporal `retrograde`.**

### Design
Turn the existing (under-used) reuse machinery into a first-class **recapitulation** feature:
- **R1. Add `retrograde()` to `transition.py`**, reverse frame order of the 139-dim clip; reverse
  L/R foot-contact channels; Gaussian-smooth (σ≈3 frames) the joins to remove velocity
  discontinuities. Unit-testable pure-numpy, like `mirror`/`retime`. (`retrograde(retrograde(x))≈x`.)
- **R2. Recapitulation routing in `story.py`.** Using `Section.repeat_of` (S3): when a later
  section repeats an earlier one, **reuse the earlier section's chosen motion** (cached), retimed
  to the new length (DTW/`retime`), optionally varied (`mirror`/`retrograde`/`amplitude`), instead
  of scoring LODGE/EDGE afresh. Concretely: reduce `_W_REUSE` cost / raise the reuse bonus when
  `repeat_of` is set so the assembler prefers recapitulation for repeated sections.
- **R3. Storyboard directive.** Update the `storyboard.py` prompt so the LLM, given `repeat_of`
  and roles, explicitly plans recapitulation and picks the variation (mirror at the return,
  retrograde for a B→A′, etc.), setting `reuse_of` + `variation` it already supports.
- **R4. "Mirror the intro at the end."** A common special case: if the outro shares the intro's
  label (or the user asks), reuse the intro clip **mirrored/retrograded** as the outro, a direct,
  perceptually strong ABA close.
- **R5. Structuredness metric**, add **Section-Repetition-Correlation (SRC)** to
  `story_metrics.py`: pose-feature cosine similarity between motion of same-label sections
  (extends the existing `motif_recurrence`), so we can measure that the dance mirrors the music's
  ABA form. (See §6.)

---

## 3. Best-of-K seed sampling for beat alignment (Thread 3)

### Findings
- **Both EDGE and LODGE are diffusion models and fully seed-stochastic**, different initial noise
  `z_T` gives independent, differently-beat-aligned dances for the same music. Neither explicitly
  optimizes beat alignment, so **BAS variance across seeds is substantial (~±0.03–0.06)** → picking
  the best of several is effective.
- **Best-of-N is the proven, low-risk approach** (Ma et al. 2025, *Inference-Time Scaling for
  Diffusion*, arXiv:2501.09732: random best-of-N often beats fancier search). Recommended **K=4–8**
  seeds/segment; use more (8) for rhythmically complex sections, fewer (4) for calm ones. Samples
  are independent → embarrassingly parallel.
- **Beat Alignment Score (BAS)** (AIST++/Li et al. 2021): motion beats = local minima of joint
  velocity; `BAS = mean_k exp(−‖t^motion_k − nearest_music_beat‖² / 2σ²)`, σ≈3 frames @30fps.
  Complement with **PFC** (EDGE foot-contact plausibility) and energy-match. SOTA reference for
  explicit beat modeling: **Beat-It** (ECCV 2024). Follow-up: **Universal Guidance**
  (arXiv:2302.07121) to steer sampling toward beat alignment with no retraining.

### Design
- **B1. Add a beat-alignment metric first** (`agentlodge/dance/beat_metrics.py`): implement BAS
  (motion kinematic beats vs `librosa` music beats) + optionally PFC. This finally *measures* the
  weakness and is a prerequisite for everything else. Wire it into `story_metrics.py` /
  `pipeline_log.json`.
- **B2. Best-of-K generation module.** For a given section, request **K seeded samples** from the
  generator subprocess (LODGE/EDGE already take a seed; expose `--seed`/`torch.manual_seed`), score
  each with `w1·BAS + w2·PFC + w3·energy_match` (start `w1=0.6`), and return the argmax candidate to
  the assembler. Cache candidates for reuse/editing.
- **B3. Targeted, not global.** Run best-of-K only where it pays: the **agentically-chosen**
  section for each slot, and sections flagged as high-tempo/onset-dense. This directly strengthens
  "the agentically-chosen dances" the user cares about while bounding compute.
- **B4. (Later) Universal Guidance** using a differentiable BAS on the Tweedie estimate, as a
  higher-ceiling follow-up once best-of-K is validated. Watch for verifier-hacking (degenerate
  high-BAS motion).

---

## 4. LLM text descriptions of dance segments (Thread 4)

### Findings
- **Text↔motion models exist**: **TMR** (Petrovich et al., ICCV 2023, open-source) gives a
  cross-modal cosine similarity between a text description and a motion clip, usable directly as a
  **verifier/critic**. **MotionGPT / TM2T / MotionLLM** can *caption* a clip in natural language.
- Cheap baseline without a learned model: **kinematic-feature captions** derived from what we
  already compute (energy, velocity profile, foot contacts, spatial extent, periodicity,
  symmetry) → a templated sentence per segment.
- LLM-reasoning literature (CoT, self-consistency, ReAct, Reflexion) consistently shows
  **intermediate natural-language representations improve decision quality**, i.e., letting the
  storyboard LLM reason over *descriptions* rather than raw energy numbers should help selection.

### Design
- **D1. Segment captioner** (`agentlodge/agent/segment_caption.py`): start with a **kinematic
  templated caption** (no new heavy model), e.g. "high-energy, wide traveling movement with sharp
  accents on the beat, symmetric arms." Optional upgrade: a learned captioner (MotionGPT/TM2T).
- **D2. Description-grounded selection.** Feed each candidate's caption into the storyboard/
  selection prompt so the LLM compares *described* options; this also makes `vocabulary` (currently
  inert, see prior analysis) *meaningful* by matching plan-vocabulary to candidate captions.
- **D3. Plan↔realization verifier.** Use **TMR** cosine similarity between the plan's intended
  description and the realized segment as a numeric check that the assembly matched intent; surface
  it in metrics and reuse it as the critic in Thread 5.

---

## 5. Natural-language editing agent + verification loop (Thread 5)

### Findings
- **"Nano Banana"** = Gemini's native image gen; **"AgentBanana"** = the engineering pattern that
  wraps it in a **propose → apply → verify → refine** loop (planner/parser → editor → VLM/LLM
  critic → stop-condition, typically **3–5 iterations**). Backed academically by **Self-Refine**
  and **Reflexion** (both NeurIPS 2023). Caveat: self-correction has limits (Huang et al. 2023) -
  keep the loop bounded and use an *objective* verifier where possible.
- **NL motion editing prior art**: MotionFix (text-based 3D motion editing), Goel et al.'s
  LLM-driven **iterative motion editing via programs** (SIGGRAPH 2024), EDGE's editing/inpainting,
  DNO/OmniControl. Verification via **TMR** (text-motion critic) + kinematic-metric deltas.

### Design, an editing agent over AgentLODGE's *own* bounded operations
The pipeline already has a small, safe operation set. Map NL → those ops, apply, verify, loop:
- **E1. Instruction parser (LLM).** Map a request to a bounded op set already available:
  re-select segment source (LODGE↔EDGE), **re-sample K seeds** (Thread 3), `retime`,
  `mirror`, `retrograde` (Thread 2), `amplitude_scale`, motif reuse / recapitulate, change a
  section's `target_intensity`/`generator_bias`, or re-run assembly. Resolve time references
  ("at 0:30", "the chorus") against the detected `structure`.
- **E2. Apply** the op to the cached per-section motion (no full regen needed for most edits).
- **E3. Verifier / critic.** Check the edit achieved intent using *objective* signals first:
  metric deltas (energy↑ for "more energetic", **BAS↑** for "tighten the beat", detected
  spin/jump for "add a spin"), plus **TMR** text-motion alignment, plus optionally a **VLM watching
  the re-rendered clip**. Return `{satisfied, feedback}`.
- **E4. Bounded refine loop** (max ~3–5 iters, Self-Refine/Reflexion style): if not satisfied, feed
  the critic's feedback back to the parser to adjust the op (e.g., increase amplitude, raise K),
  else stop; on non-convergence return the best attempt + a clear explanation of what couldn't be
  achieved. Keep a Reflexion-style memory of "what worked" across turns.
- **E5. Interface.** A conversational `edit_dance(instruction, state)` entrypoint; the agent keeps
  the current dance + per-section cache as state so edits are incremental.

---

## 6. Cross-cutting: metrics

Add the missing measurements so improvements are quantifiable (feeds every thread + the editor's
verifier):
- **Beat Alignment Score (BAS)** and **PFC**, new `beat_metrics.py` (Thread 3, prerequisite).
- **Section-Repetition-Correlation (SRC)**, same-label motion similarity (Thread 2).
- **Energy-Arc Correlation (EAC)**, already ≈ `arc_adherence`; keep.
- **Plan↔realization TMR alignment**, Thread 4 verifier.
- Optional aggregate **Choreographic Structure Score**: `CSS = α·BAS + β·SRC + γ·EAC + δ·PFC`.

---

## 7. Proposed architecture (mapped to modules)

```
audio ─▶ structure.py            (S1 spectral-cluster labels, S2 hypermeter, S3 repeat_of)
         │
         ▼
      storyboard.py              (R3 recapitulation directives; D2 reason over segment captions)
         │  plan (arc + per-section: role, intensity, bias, reuse/variation)
         ▼
      story.py  ◀── best_of_k.py (B2/B3 K-seed candidates per chosen slot, scored by beat_metrics)
         │        ◀── segment_caption.py (D1 caption each candidate)
         │        R2 recapitulation reuse + R1 retrograde / mirror / retime variation
         ▼
   assembled "story" dance ─▶ story_metrics.py (+ beat_metrics.py, SRC, TMR alignment)
         │
         ▼
      edit_agent.py  (E1 parse → E2 apply bounded op → E3 verify (BAS/TMR/VLM) → E4 loop)
         ▲ natural-language user instructions
```

New modules: `dance/beat_metrics.py`, `dance/best_of_k.py`, `agent/segment_caption.py`,
`agent/edit_agent.py`; new primitive `transition.retrograde`; extensions to `structure.py`,
`storyboard.py`, `story.py`, `story_metrics.py`.

---

## 8. Phased roadmap

**Phase 0, Measure the weakness (fast, high-value).** ✅ **DONE** (commit 561e60f)
`beat_metrics.py` (BAS + beat coverage + FK-free foot-contact consistency) wired into
`compute_story_metrics` + `pipeline_log`. Beat alignment is now visible and every later change is
measurable. Also landed `transition.retrograde` (Phase-1 primitive) + wired into the motif-reuse
path + storyboard schema. 9 new tests (21 total pass). **Song-150 baseline:** BAS LODGE 0.406 /
EDGE 0.391 / STORY 0.408; foot-consistency LODGE 0.513 / EDGE 1.000 / STORY 0.786. *No model work.*

**Phase 1, Structure & recapitulation (mostly training-free, high payoff).** ✅ **DONE** (af15561)
`Section.repeat_of` + normalized-Laplacian **spectral labels** (opt-in `AGENTLODGE_STRUCTURE_SPECTRAL`)
+ `retrograde` primitive + **recapitulation** close (`AGENTLODGE_STORY_RECAP`: reuse the opening
mirrored+retrograded at the final section) + storyboard directive + **SRC** metric. Pure-numpy + tested.

**Phase 2, Best-of-K for beat alignment.** ✅ **DONE** (1387e20 + integration)
`dance/best_of_k.py`, generator-agnostic best-of-K: a `generate_fn(seed)` closure, scored by
composite BAS(0.6)+foot(0.25)+energy(0.15), returns the argmax + per-seed report. **Now fully wired
into the pipeline:** seed control end-to-end (`generate_lodge_dance`/`generate_edge_dance` +
`run_lodge_inference.py`/`run_edge_inference.py --seed` + `subprocess_runner`), a `best_of_k_job`
orchestrator + `_lodge/_edge_score_transform`, and `AGENTLODGE_BEST_OF_K` (generate K seeded
whole-song candidates per generator, keep the most beat-aligned). Selection validated on real
motions + unit tests. *Live seeded-generation run needs the pod's model env (torch + checkpoints)
restored, the migrated pod lost them.*

**Phase 3, Description-grounded reasoning.** ✅ **DONE** (b1e816f)
`agent/segment_caption.py`, FK-free kinematic captions + vocabulary↔energy match (makes `vocabulary`
meaningful) + `plan_realization_alignment` verifier (optional TMR hook). Surfaced per section in the
assembler's decisions. *Optional upgrade:* learned captioner (MotionGPT/TM2T) + TMR critic.

**Phase 4, Natural-language editing agent.** ✅ **DONE** (218e69c)
`agent/edit_agent.py`, parse NL → bounded `EditOp` → apply → re-assemble → verify → bounded refine
loop over existing controls (recapitulate / set_intensity / set_bias / mirror·retrograde·amplitude /
beat), plus per-section `post_variations` in the assembler. Tested offline. *Optional:* VLM critic.

**Phase 5 (stretch), Guidance & learned components.**
Universal-Guidance beat steering (B4); learned captioner (MotionGPT); TMR fine-tuned on dance.

**Phase 6, Curated common-motion choreography.** ✅ **DONE**
`SectionPlan.common_motions` exposes the live 20-motion manifest to the whole-song LLM planner.
Plans carry sparse beat-aware cues with intensity, direction, repetition, motif identity, and
rationale. The assembler composes cues without changing section duration, caches composed sections
for exact chorus/hook recurrence, and prevents double application on reused sections. Invalid or
invented cues are rejected against the manifest; failed readability at an overly compressed beat
first retries at full intensity, then relaxes beat compression while preserving the planned
section position. The fresh-pod `make_song_bestofk.py` path now passes the runtime OpenAI key and
records storyboard and cue-level section scores.

Suggested order: **Phase 0 → 1 → 2** give the biggest structural + beat-alignment wins with the
least risk; 3–4 add the reasoning/interactivity the user asked for.

---

## 9. References (selected; full per-thread reviews preserved separately)

**Music structure & theory:** Foote (ICME 2000); McFee & Ellis, *Spectral clustering* (ISMIR 2014);
Serrà et al. (AAAI 2012); Nieto & Mysore, *MSAF* (ISMIR 2016); Müller, *Fundamentals of Music
Processing* (2015); Lerdahl & Jackendoff, *GTTM* (1983); Huron, *Sweet Anticipation* (2006);
Dhariwal et al., *Jukebox* (arXiv:2005.00341).

**Choreographic composition:** Blom & Chaplin, *The Intimate Act of Choreography* (1982); Humphrey,
*The Art of Making Dances* (1959); Smith-Autard, *Dance Composition* (2019); Laban, *Choreutics*
(1966); Aristidou et al., *Rhythm is a Dancer* (IEEE TVCG 2022, arXiv:2111.12159);
*Music-to-Dance via Atomic Movements* (2025).

**Dance generation:** Li et al., *AI Choreographer / AIST++* (ICCV 2021, arXiv:2101.08779);
Li et al., *Bailando* (CVPR 2022, arXiv:2203.13055); Tseng et al., *EDGE* (CVPR 2023,
arXiv:2211.10658); Li et al., *LODGE* (CVPR 2024, arXiv:2403.10518) & *LODGE++* (arXiv:2410.20389);
Li et al., *FineDance* (ICCV 2023, arXiv:2212.03741); Huang et al., *Beat-It* (ECCV 2024,
arXiv:2407.07554).

**Diffusion sampling & beat:** Ho & Salimans, *CFG* (arXiv:2207.12598); Bansal et al., *Universal
Guidance* (CVPR 2023, arXiv:2302.07121); Chung et al., *DPS* (ICLR 2023, arXiv:2209.14687);
Ma et al., *Inference-Time Scaling for Diffusion* (arXiv:2501.09732); Snell et al., *Scaling
Test-Time Compute* (arXiv:2408.03314).

**Text↔motion / reasoning:** Petrovich et al., *TMR* (ICCV 2023, arXiv:2305.00976); Jiang et al.,
*MotionGPT* (NeurIPS 2023, arXiv:2306.14795); Guo et al., *TM2T* (ECCV 2022, arXiv:2207.01696);
Chen et al., *MotionLLM* (arXiv:2405.20340); Tevet et al., *MDM* (ICLR 2023, arXiv:2209.14916).

**Agentic editing / self-refinement:** Madaan et al., *Self-Refine* (NeurIPS 2023, arXiv:2303.17651);
Shinn et al., *Reflexion* (NeurIPS 2023, arXiv:2303.11366); Athanasiou et al., *MotionFix* (2024);
Goel et al., *Iterative Motion Editing* (SIGGRAPH 2024); Google, *Gemini native image (“Nano
Banana”)* docs.
