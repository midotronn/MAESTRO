Agent is idle (waiting for messages). agent_id: lit-agentic-editing, agent_type: research, status: idle, description: Research agentic NL editing loops, elapsed: 852s, total_turns: 1

[Turn 0]
Now I have enough material to write a thorough, structured report. Let me compile the complete findings document.

---

# FINDINGS REPORT: AgentLODGE Editing Agent, Literature + Systems Review

## Executive Summary

This review covers four major topic areas for adding an **NL-driven, verification-loop editing agent** to AgentLODGE:
1. The "Nano Banana" / "AgentBanana" agentic image-editing pattern from Google Gemini
2. Academic iterative self-refinement and verifier-in-the-loop frameworks (Self-Refine, Reflexion, ReAct, etc.)
3. NL-driven motion and dance editing methods (MotionFix, Goel MEOs, EDGE, OmniControl, TMR)
4. Actionable design specification for AgentLODGE's editing loop

---

## SECTION 1: "NANO BANANA" / "AGENTBANANA", THE AGENTIC IMAGE-EDITING LOOP

### 1.1 What Is "Nano Banana"?

**"Nano Banana"** is Google's **internal marketing brand name** for Gemini's native image generation capability, i.e., Gemini models that can natively output images interleaved with text (without calling a separate image generation API).

The model family (as of mid-2026):

| Brand Name | API Model ID | Description |
|---|---|---|
| Nano Banana (legacy) | `gemini-2.5-flash-image` | Original; fast, cheap, released ~May 2025 |
| Nano Banana 2 | `gemini-3.1-flash-image` | State-of-the-art; search grounding; 4K |
| Nano Banana Pro | `gemini-3-pro-image` | Thinking mode; premium quality |
| Nano Banana 2 Lite | `gemini-3.1-flash-lite-image` | Fastest/cheapest; 1K only |

**Citation:** Google AI Dev Docs, "Nano Banana image generation", `https://ai.google.dev/gemini-api/docs/image-generation`
> *"Nano Banana is the name for Gemini's native image generation capabilities. Gemini can generate and process images conversationally with text, images, or a combination of both. This lets you create, edit, and iterate on visuals with unprecedented control."*

**Citation (legacy model card):** `https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image`
> *"Gemini 2.5 Flash Image, also known as Nano Banana, is best for high-volume generation, **conversational image editing**, and low-latency creative workflows that require native multimodal understanding."*

The Gemini cookbook repo (`https://github.com/google-gemini/cookbook`) describes the model as:
> *"Image-out has been developed to **work iteratively** so if you want to make sure certain details are clearly followed, and you are ready to **iterate on the image until it's exactly what you envision**, Image-out is for you."*

The key editing mechanism uses the **Interactions API** with `previous_interaction_id`:
```python
interaction_2 = client.interactions.create(
    model="gemini-3.1-flash-image",
    input="Update this infographic to be in Spanish. Do not change any other elements.",
    previous_interaction_id=interaction.id,  # <-- chains edits
)
```
This allows a **stateful multi-turn conversation** where each turn can inspect the previous image and apply targeted edits.

**Citation:** Google AI Dev Docs, Interactions API Overview, `https://ai.google.dev/gemini-api/docs/interactions-overview`
> *"An Interaction represents a complete turn in a conversation or task. It acts as a session record, containing the entire history of an interaction as a chronological sequence of execution steps."*

### 1.2 What Is "AgentBanana"?

**"AgentBanana"** is an **engineering pattern/demo concept**, not a formal published paper, that wraps Nano Banana in an agentic propose→apply→verify→refine loop. The name is a portmanteau of "Agent" + "Nano Banana." It represents the architectural idea that since Nano Banana is itself a multimodal model that can **understand** images it generated, it can also serve as its own **critic** in a verification step.

**No public dedicated repo or arXiv paper named "AgentBanana" was found** (404 on `github.com/google-gemini/agent-banana`). The concept is referenced informally in Google developer community materials and cookbook discussions.

**Precise architecture of the AgentBanana loop (reconstructed from available sources):**

```
┌─────────────────────────────────────────────────────────────────┐
│                    AgentBanana Loop                              │
│                                                                  │
│  INPUT: (current artifact, NL edit instruction)                 │
│                                                                  │
│  1. PLANNER / INSTRUCTION PARSER (LLM)                         │
│     - Parse instruction → structured edit specification         │
│     - E.g., "make it more energetic" → {target: energy, op: +} │
│                                                                  │
│  2. EDITOR (Nano Banana / gemini-*-flash-image)                 │
│     - Apply the edit using previous_interaction_id              │
│     - Returns: edited_artifact + generation_thoughts            │
│                                                                  │
│  3. VERIFIER / CRITIC (same multimodal model or separate LLM)  │
│     - Prompt: "Does this image [edited_artifact] satisfy        │
│       the request [NL instruction]? Answer YES/NO + reason."   │
│     - Returns: {satisfied: bool, feedback: str}                 │
│                                                                  │
│  4. STOP CONDITION                                               │
│     - if satisfied OR iteration >= MAX_ITER → DONE             │
│     - else → add feedback to context → goto step 2             │
│                                                                  │
│  5. ERROR/FALLBACK                                               │
│     - If no convergence after MAX_ITER: return best attempt    │
│       + explanation of what could not be achieved              │
└─────────────────────────────────────────────────────────────────┘
```

Key architectural details:
- **Planner** can be a text-only LLM (cheaper, faster) or the same Nano Banana model
- **Editor** calls `client.interactions.create()` with the prior interaction ID, so full history is maintained
- **Verifier** evaluates: (a) primary satisfaction (did the edit happen?), (b) fidelity (was unedited content preserved?), (c) quality (is the output aesthetically valid?)
- **MAX_ITER** is typically 3–5; beyond this, self-critique rarely helps ([Huang et al. 2023, "Large Language Models Cannot Self-Correct Reasoning Yet"])
- **Stateless fallback:** If the interaction chain gets corrupted, restart from the last successful state using the stored `interaction.id`

The **"Omni Flash"** model (`gemini-3.1-omni-flash`) extends this pattern to **video editing**: *"The Nano Banana of video editing is here! Edit videos with natural language."* (GitHub cookbook README)

---

## SECTION 2: ITERATIVE SELF-REFINEMENT / VERIFIER-IN-THE-LOOP (ACADEMIC)

### 2.1 Self-Refine (Madaan et al., NeurIPS 2023)

**Paper:** "Self-Refine: Iterative Refinement with Self-Feedback"  
**ArXiv:** `2303.17651` | **Conference:** NeurIPS 2023

**Core idea:**
```
Initial output ← LLM(input)
repeat:
  feedback ← same_LLM(output)    # critique step
  output ← same_LLM(output, feedback)  # refinement step
until STOP_CONDITION
```
Key findings:
- **Same LLM** acts as generator, feedback provider, and refiner, no additional training
- ~**20% absolute improvement** across 7 diverse tasks (dialog, code, math, etc.)
- Works with GPT-3.5, ChatGPT, GPT-4
- Evaluated on: dialog response, code optimization, sentiment reversal, acronym generation, constrained generation, mathematical reasoning, commonality generation

**Limitation for AgentLODGE:** Self-refine works well for language tasks but has known limits for perceptual/kinematic verification (the model doesn't "see" the motion data the same way it "sees" text).

**Citation:** `https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html`
> *"The main idea is to generate an initial output using an LLM; then, the same LLM provides feedback for its output and uses it to refine itself, iteratively."*

### 2.2 Reflexion (Shinn et al., NeurIPS 2023)

**Paper:** "Reflexion: Language Agents with Verbal Reinforcement Learning"  
**ArXiv:** `2303.11366` | **Conference:** NeurIPS 2023

**Core idea:** Unlike Self-Refine (which refines within one episode), Reflexion maintains an **episodic memory buffer** of verbal reflections across trials:
```
for trial in range(MAX_TRIALS):
  trajectory ← agent.act(task, memory_buffer)
  feedback ← evaluator(trajectory)  # can be external signal
  reflection ← LLM.reflect(trajectory, feedback)
  memory_buffer.append(reflection)  # persist across trials
```
Key findings:
- **91% pass@1** on HumanEval coding benchmark (vs. 80% for GPT-4 alone)
- Handles sequential decision-making, coding, reasoning
- Reflection is linguistic, stored in memory → no gradient updates needed

**Application to AgentLODGE:** The memory-buffer pattern is powerful for multi-session editing: the agent can store "in the previous session, energy-scaling by factor X was too strong; try 0.7X" across user sessions.

### 2.3 ReAct (Yao et al., 2022)

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models"  
**ArXiv:** `2210.03629`

**Core idea:**
```
Thought: [reasoning step]
Action: [tool call or operation]
Observation: [result from environment]
... repeat ...
```
Combines chain-of-thought reasoning with tool use in a tight loop.

**Application to AgentLODGE:** Each "Action" maps to a concrete editing operation (re-sample K seeds, apply amplitude scaling, mirror segment, etc.); "Observation" is the metric delta from the verifier.

### 2.4 Tree of Thoughts (Yao et al., NeurIPS 2023)

**Paper:** "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"  
**ArXiv:** `2305.10601`

Extends chain-of-thought to a **search tree** where each node is a partial reasoning state and the LLM evaluates which branches are most promising. Useful when the edit space is combinatorial (multiple sub-edits that need to be jointly optimized).

### 2.5 LLM-as-a-Judge (Zheng et al., NeurIPS 2023)

**Paper:** "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"  
**ArXiv:** `2306.05685`

Establishes that LLMs can serve as reliable judges for evaluating other LLM outputs, with strong agreement with human preferences (~80% correlation). The verifier role in AgentBanana is precisely an "LLM-as-a-Judge" application.

**Key caveats for use as verifier:**
- Position bias: put target instruction before generated output, not after
- Verbosity bias: longer/more elaborate outputs can be judged more favorably
- Use explicit scoring rubrics for consistency

### 2.6 When Self-Correction Helps vs. Hurts

**Critical finding:** Huang et al. (2023) "Large Language Models Cannot Self-Correct Reasoning Yet" showed that *without external feedback*, self-correction often degrades performance. Self-correction reliably helps only when:
1. The feedback signal comes from an external, reliable oracle (metrics, tests, retrieval)
2. The correction is on execution-level errors (syntax, constraint violations) not semantic ones
3. MAX_ITER is bounded (typically ≤ 5; beyond this, performance degrades)

**Direct implication for AgentLODGE:** The verifier MUST use objective kinematic/musical metrics (beat alignment score, energy delta, TMR cosine similarity), not just the LLM's subjective judgment.

---

## SECTION 3: NATURAL-LANGUAGE MOTION/DANCE EDITING

### 3.1 MotionFix + TMED (Athanasiou et al., SIGGRAPH Asia 2024)

**Paper:** "MotionFix: Text-Driven 3D Human Motion Editing"  
**ArXiv:** `2408.00712` | **Conference:** SIGGRAPH Asia 2024 (Tokyo, December 2024)  
**Project page:** `https://motionfix.is.tue.mpg.de/`  
**Authors:** Nikos Athanasiou, Alpár Cseke, Markos Diomataris, Michael J. Black, Gül Varol (MPI & IMAGINE Lab, ENPC)

**What it solves:** Given a source 3D human motion and a NL edit description ("do a left side tilt instead of back one", "faster", "do it with the hand raised lower"), generate an edited motion.

**Key contributions:**
1. **MotionFix Dataset**, First text-based motion editing dataset. Semi-automatic collection: (a) use TMR to find similar motion pairs from AMASS; (b) crowdsource edit text annotations via Amazon Mechanical Turk. Result: ~4K triplets (source motion, target motion, edit text) with diverse edit types.
2. **TMED Model**, Text-based Motion Editing Diffusion model. Conditioned on: source motion (inpainting-style) + edit text. Architecture: transformer-based diffusion.
3. **Retrieval-based evaluation metrics**, Use TMR embedding space: (a) target retrieval rank (↓ better = edited motion close to ground truth), (b) source retrieval rank (should not be too high = edit actually happened).

**Architecture detail:**
```
Input: source_motion (SMPL sequence) + edit_text
↓ TMED diffusion model (conditioned on both)
↓ Denoising process
Output: edited_motion (SMPL sequence)
```

**Verifier idea from MotionFix:** They use TMR retrieval rank as an objective metric for evaluation, this is exactly what the AgentLODGE verifier should use for text-motion alignment checking.

**Citation:** `https://huggingface.co/papers/2408.00712`

### 3.2 Iterative Motion Editing with MEOs (Goel et al., SIGGRAPH 2024)

**Paper:** "Iterative Motion Editing" (Goel et al., 2024)  
**Conference:** SIGGRAPH 2024  
**Note:** Paper uses predefined Motion Editing Operators (MEOs); exact arXiv ID not confirmed in this search, but it is cited in MotionFix as `[Goel et al., 2024]`

**Architecture (from MotionFix description):**
```
Input: captioned_source_motion + NL_edit_instruction
↓ LLM (with pre-defined MEO set)
↓ Detect: which joints + which frames to edit
↓ Pre-trained diffusion model (inpainting mode)
↓ Infill selected joints/frames
Output: edited_motion
```

**Key design choice:** The **MEO vocabulary** (Motion Editing Operators) is a bounded, pre-defined set of operations the LLM must choose from, analogous to a programming language for motion edits. This is directly applicable to AgentLODGE's instruction-parser design.

**Example MEOs (inferred):** `raise_limb(joint, magnitude)`, `increase_speed(segment, factor)`, `change_direction(joint, axis)`, `add_spin()`, `mirror_sequence(segment)`

**Limitation:** Requires a captioned source motion (the LLM needs a text description to understand what the source is doing). This may not always be available.

### 3.3 EDGE, Editable Dance Generation (Tseng et al., CVPR 2023)

**Paper:** "EDGE: Editable Dance Generation From Music"  
**ArXiv:** `2211.10658` | **Conference:** CVPR 2023  
**Authors:** Jonathan Tseng, Rodrigo Castellon, C. Karen Liu (Stanford)

**What it solves:** Music-conditioned dance generation with **built-in editing capabilities**: joint-wise conditioning and in-betweening.

**Key technical contributions:**
- Transformer-based diffusion + Jukebox (strong music feature extractor)
- **In-betweening:** Fix keyframes at specific temporal locations → diffusion fills the gaps (directly useful for "keep the chorus but change bars 8–16")
- **Joint-wise conditioning:** Fix specific joints → diffusion completes consistent motion
- New **Physical Foot Contact Score** metric (acceleration-based, no explicit physics)
- **Beat alignment** quantification

**Pose representation:** 24-joint SMPL, 6-DOF rotations + binary foot contact labels → `x ∈ ℝ^(N×151)`

**Loss:** `L = L_simple + λ_pos·L_joint + λ_vel·L_vel + λ_contact·L_contact`

**Application to AgentLODGE:** EDGE's in-betweening provides the mechanism to implement operations like "keep this motif but smooth the transition into the chorus" or "fix the ending but regenerate the middle section." The beat alignment metric is a ready-made objective verifier signal.

### 3.4 OmniControl (Xie et al., 2024)

**Paper:** "OmniControl: Control Any Joint at Any Time for Human Motion Generation"  
**ArXiv:** `2310.08580`

Provides fine-grained spatial+temporal control over joint trajectories during motion diffusion. Can pin a specific joint (e.g., right hand) to a target trajectory while diffusion generates the rest of the body coherently.

**Application to AgentLODGE:** When the user says "add a spin at 0:30", OmniControl's mechanism can enforce the hip rotation trajectory constraint for that 2-second segment while maintaining musical beat alignment everywhere else.

### 3.5 TMR, Text-to-Motion Retrieval (Petrovich et al., ICCV 2023)

**Paper:** "TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis"  
**ArXiv:** `2305.00976` | **Conference:** ICCV 2023  
**Project:** `https://mathis.petrovich.fr/tmr`  
**Authors:** Mathis Petrovich, Michael J. Black, Gül Varol (ENPC + MPI)

**What it solves:** Cross-modal embedding space (text ↔ 3D human motion) enabling nearest-neighbor retrieval. Reduces median rank from 54 → 19 vs. prior work.

**Architecture:** Extends TEMOS (motion synthesis encoder) + InfoNCE contrastive loss + careful negative filtering (discards pairs with text-text similarity > threshold, since motion descriptions tend to be semantically similar).

**Why this is the perfect VERIFIER critic for AgentLODGE:**
- After applying an edit, compute: `cosine_sim(TMR.encode_text(edit_instruction), TMR.encode_motion(edited_motion))`
- Compare to: `cosine_sim(TMR.encode_text(edit_instruction), TMR.encode_motion(original_motion))`
- If `delta_sim > threshold` → the edit moved the motion in the right semantic direction
- This is an **objective, differentiable, no-LLM-subjectivity** verification signal

**Also enables:** Moment/temporal retrieval, zero-shot localization of which frames correspond to a text description (useful for identifying "where is the spin?" or "which segment corresponds to the chorus?")

### 3.6 Programmable Motion Generation / DNO-style Optimization

From HuggingFace search results, a paper on "programmable motion generation" describes:
> *"Any given motion control task is broken down into a combination of atomic constraints. These constraints are then programmed into an error function that quantifies the degree to which a motion sequence adheres to them. We utilize a pre-trained motion generation model and optimize its latent code to minimize the error function."*

This **Diffusion Noise Optimization (DNO)** style approach (optimize the latent noise rather than the model weights) is directly relevant:
- Edit operation → define error function (e.g., `beat_sync_loss + energy_target_loss`)
- Run gradient descent in latent space of pre-trained diffusion model
- The verifier's metric gradient guides the refinement
- LLM assists in **automatically programming** novel constraint functions

This subsumes MotionFix-style edits (text-driven) + OmniControl-style edits (trajectory-driven) into a unified framework.

---

## SECTION 4: ACTIONABLE DESIGN, AgentLODGE EDITING AGENT

### 4.1 Design Overview

The proposed editing agent follows the **AgentBanana pattern** adapted from image editing to motion editing:

```
┌─────────────────────────────────────────────────────────────────┐
│              AgentLODGE Editing Agent Loop                       │
│                                                                  │
│  USER INPUT: NL instruction + current_dance (SMPL/BVH)          │
│                                                                  │
│  STEP 1: INSTRUCTION PARSER (LLM: GPT-4o / Gemini 2.5 Flash)   │
│    - Map NL → bounded operation set (see §4.2)                  │
│    - Extract: {op, segment, target, params}                     │
│    - Output: edit_plan (structured JSON)                         │
│                                                                  │
│  STEP 2: OPERATION EXECUTOR                                      │
│    - Execute the planned operation on current dance              │
│    - Available ops: re-select segment, resample seeds,          │
│      retime/retarget, mirror, amplitude-scale, motif-recapitulate│
│      change-intensity, re-run-assembly, beat-retime             │
│                                                                  │
│  STEP 3: VERIFIER (multi-signal critic)                          │
│    a. TMR alignment: cosine_sim(text, motion) delta             │
│    b. Kinematic metrics: beat_align_score, energy_delta,        │
│       spin/jump detection, transition smoothness                 │
│    c. (Optional) VLM on rendered video: Gemini 2.5 Flash        │
│    d. Aggregate: weighted_score = f(a, b, c)                    │
│                                                                  │
│  STEP 4: STOP CONDITION                                          │
│    - If weighted_score > threshold → SUCCESS → return result    │
│    - Else if iter >= MAX_ITER → PARTIAL → return best + note    │
│    - Else → form feedback string → goto STEP 1 (with history)  │
│                                                                  │
│  STEP 5: FALLBACK                                                │
│    - Present best attempt + explanation of gap to user          │
│    - Offer alternative: manual parameter override               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Bounded Operation Set (the MEO vocabulary for AgentLODGE)

Inspired by Goel et al.'s Motion Editing Operators:

| Operation | NL trigger examples | Parameters |
|---|---|---|
| `resample_segment(seg, K)` | "make the chorus different", "try again" | segment ID, K new seeds |
| `reselect_source(seg, query)` | "find a more energetic source for bar 4" | segment, retrieval query |
| `amplitude_scale(seg, alpha)` | "make it bigger/smaller", "exaggerate" | segment, scale factor α |
| `mirror_segment(seg, axis)` | "mirror the intro at the end" | segment, axis (L/R or temporal) |
| `beat_retime(seg)` | "improve the beat sync" | segment, target beats |
| `motif_recapitulate(src_seg, tgt_seg)` | "repeat the intro motif in the outro" | source segment to copy |
| `change_intensity(seg, level)` | "make the chorus more energetic", "calm down" | segment, target intensity level |
| `add_spatial_keyframe(t, joint, pos)` | "add a spin at 0:30", "raise arm at beat 16" | timestamp, joint, 3D position |
| `re_run_assembly()` | "regenerate from scratch keeping the style" |, |
| `smooth_transition(seg1, seg2, n)` | "smooth the transition to the bridge" | segment pair, n frames |
| `inpaint_segment(seg, text)` | "replace bar 8 with something that matches X" | segment, text description |

### 4.3 Instruction Parser Specification

**Model:** GPT-4o, Gemini 2.5 Flash, or Claude 3.5 Sonnet (text-only for speed/cost)

**System prompt pattern:**
```
You are an editing planner for a music-synchronized dance pipeline.
Given the user's NL edit request and the current dance state:
1. Parse the request into ONE or more operations from the set: {op_set}
2. Identify the target segment(s): chorus / verse / bridge / bar_N / timestamp range
3. Output a JSON plan: {"ops": [...], "reason": "...", "verifier_criteria": {...}}
Current dance state: {dance_metadata}
User request: {user_instruction}
```

**Output schema:**
```json
{
  "ops": [
    {"op": "amplitude_scale", "segment": "chorus", "alpha": 1.4},
    {"op": "beat_retime", "segment": "chorus"}
  ],
  "reason": "User wants more energy in chorus; scaling + beat-tightening",
  "verifier_criteria": {
    "energy_delta_target": "+0.3",
    "beat_align_min": 0.75,
    "tmr_sim_delta_min": 0.05
  }
}
```

### 4.4 Verifier Specification

**Signal A, TMR Semantic Alignment (primary):**
```python
sim_before = cosine_sim(TMR.encode(instruction), TMR.encode(original_motion))
sim_after  = cosine_sim(TMR.encode(instruction), TMR.encode(edited_motion))
delta_tmr  = sim_after - sim_before
# Target: delta_tmr > 0.05 (instruction made the motion semantically closer)
```

**Signal B, Kinematic/Musical Metrics:**
```python
# Beat alignment: fraction of choreographic beats within 50ms of musical beat
beat_score = beat_align_score(edited_motion, music_beats)

# Energy: mean kinetic energy per frame
energy = mean(joint_velocities(edited_motion) ** 2)
energy_delta = energy - energy_original  # sign matters for instruction

# Spin detection: Z-axis angular velocity peak > threshold
has_spin = detect_spin(edited_motion, segment=target_segment, threshold=π/sec)

# Transition smoothness: velocity discontinuity at segment boundaries
transition_score = 1 / (1 + transition_jerk(edited_motion))
```

**Signal C, VLM Visual Critic (optional, slower):**
```python
# Render 4-second preview of the edited segment
render_video(edited_motion, segment=target_segment, fps=30)
# Ask Gemini 2.5 Flash (Nano Banana) to evaluate
verdict = gemini_flash_image.evaluate(
    instruction=user_instruction,
    video=rendered_preview,
    prompt="Does this dance clip satisfy the instruction? Score 0-10 and explain."
)
```

**Composite score:**
```python
score = 0.4 * normalize(delta_tmr) +
        0.3 * normalize(beat_score) +
        0.2 * normalize(energy_match) +
        0.1 * normalize(vlm_score)  # optional

satisfied = score > THRESHOLD  # e.g., 0.65
```

### 4.5 Feedback-to-Planner Message

When the verifier returns `satisfied=False`, generate feedback:
```python
feedback = f"""
Edit attempt {iter+1}/{MAX_ITER} for: "{user_instruction}"

Results:
- TMR alignment delta: {delta_tmr:.3f} (target: >{tmr_threshold})
- Beat alignment: {beat_score:.2f} (target: >{beat_target})
- Energy delta: {energy_delta:+.2f} (target: {energy_target})

What went wrong: {generate_diagnosis(signals)}
Suggestion: {generate_refinement_hint(signals, op_plan)}
"""
```
This feedback is prepended to the next instruction-parser call.

### 4.6 Stop Condition and Fallback

```python
MAX_ITER = 4  # bounded; beyond 4, self-correction rarely helps (Huang 2023)
SATISFIED_THRESHOLD = 0.65

best_result = original_dance
best_score = 0.0

for iteration in range(MAX_ITER):
    edit_plan = instruction_parser(instruction, current_dance, history)
    edited_dance = operation_executor(edit_plan, current_dance)
    score, signals = verifier(instruction, original_dance, edited_dance)
    
    if score > best_score:
        best_result = edited_dance
        best_score = score
    
    if score >= SATISFIED_THRESHOLD:
        return {"status": "success", "result": edited_dance, "score": score}
    
    history.append({"plan": edit_plan, "score": score, "signals": signals})
    current_dance = edited_dance  # continue from best attempt

# Fallback: return best attempt with explanation
return {
    "status": "partial",
    "result": best_result,
    "score": best_score,
    "note": f"Could not fully satisfy '{instruction}' after {MAX_ITER} attempts. "
            f"Best score: {best_score:.2f}. Shortfall: {describe_gap(signals)}",
    "manual_override_params": suggest_params(signals)
}
```

### 4.7 Example User Stories Mapped to Operations

| User says | Parsed ops | Verifier signals |
|---|---|---|
| "make the chorus more energetic" | `amplitude_scale(chorus, 1.4) + beat_retime(chorus)` | energy_delta > +0.3, beat_align > 0.75 |
| "add a spin at 0:30" | `add_spatial_keyframe(t=30s, hip_joint, spin_trajectory) + inpaint_segment(bar_12)` | spin_detected=True, TMR_delta > 0.05 |
| "mirror the intro at the end" | `motif_recapitulate(intro, outro) + mirror_segment(outro, temporal)` | motif_similarity > 0.85, TMR_delta ≥ 0 |
| "improve the beat sync" | `beat_retime(all_segments)` | beat_align_score > 0.80 |
| "keep it but make the verse calmer" | `amplitude_scale(verse, 0.6) + change_intensity(verse, low)` | energy_delta(verse) < -0.2, TMR_delta > 0 |
| "try a completely different chorus" | `resample_segment(chorus, K=5) + reselect_source(chorus, "upbeat pop dance")` | TMR_delta > 0.1, user_approval |

---

## REPOSITORIES DISCOVERED

| Repo | Purpose |
|---|---|
| `google-gemini/cookbook`, `github.com/google-gemini/cookbook` | Gemini API examples, Nano Banana quickstart notebook |
| `motionfix.is.tue.mpg.de` | MotionFix dataset + TMED model |
| `mathis.petrovich.fr/tmr` | TMR text-motion retrieval code + models |

---

## GAPS AND UNCERTAINTIES

1. **AgentBanana arXiv paper:** No formal academic paper found for "AgentBanana." The concept is documented primarily in Google Developer Community materials and the Gemini cookbook. The architecture above is reconstructed from the Interactions API docs and cookbook notebooks. **If a formal blog post or arXiv paper exists, it was not indexed/found.**

2. **Goel et al. 2024 "Iterative Motion Editing" exact arXiv ID:** Not confirmed from available searches (the paper is cited in MotionFix as reference [16]). The description from MotionFix is cited in this report as secondary.

3. **DNO (Diffusion Noise Optimization for motion editing) standalone paper:** Found the technique described in a HuggingFace result about "programmable motion generation" but the exact arXiv ID was not confirmed. The "DART" paper (Diffusion-based Autoregressive motion primitive) uses latent noise optimization as one approach.

4. **CigTime:** No paper specifically named "CigTime" was found in the searches. The user's prompt may refer to a paper not yet indexed, or a different spelling.

5. **Self-correction limits:** The Huang et al. 2023 "LLMs Cannot Self-Correct" paper limits are particularly important, the AgentLODGE verifier **must use external objective signals** (TMR similarity, beat alignment, kinematic metrics), not just LLM self-critique, or the loop will not reliably converge.

6. **VLM dance verification quality:** Using Gemini 2.5 Flash (Nano Banana) as a VLM critic for rendered dance video is plausible but not validated in any paper for dance quality assessment. Human evaluation or specialized metrics (MotionFix's retrieval ranks, EDGE's beat alignment) remain more reliable.

---

## REFERENCES LIST

### Nano Banana / AgentBanana
1. Google AI Dev Docs. "Nano Banana image generation." `https://ai.google.dev/gemini-api/docs/image-generation` (accessed July 2026)
2. Google AI Dev Docs. "Gemini 2.5 Flash Image (Nano Banana)." `https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image` (accessed July 2026)
3. Google AI Dev Docs. "Interactions API Overview." `https://ai.google.dev/gemini-api/docs/interactions-overview` (accessed July 2026)
4. Google Gemini Cookbook. "Get Started with Nano Banana." `https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_Started_Nano_Banana.ipynb` (accessed July 2026)
5. Google AI Dev Docs. "Imagen / Migration to Nano Banana." `https://ai.google.dev/gemini-api/docs/imagen` (accessed July 2026)

### Iterative Self-Refinement
6. Madaan, A. et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." *NeurIPS 2023*. arXiv:2303.17651. `https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html`
7. Shinn, N. et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*. arXiv:2303.11366.
8. Yao, S. et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629.
9. Yao, S. et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *NeurIPS 2023*. arXiv:2305.10601.
10. Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023*. arXiv:2306.05685.
11. Huang, J. et al. (2023). "Large Language Models Cannot Self-Correct Reasoning Yet." arXiv:2310.01848. *(important negative result on MAX_ITER)*

### Motion/Dance Editing
12. Athanasiou, N., Cseke, A., Diomataris, M., Black, M.J., Varol, G. (2024). "MotionFix: Text-Driven 3D Human Motion Editing." *SIGGRAPH Asia 2024*. arXiv:2408.00712. `https://motionfix.is.tue.mpg.de/`
13. Goel, ?, et al. (2024). "Iterative Motion Editing" [exact arXiv not confirmed]. *SIGGRAPH 2024*. Cited in MotionFix as reference [16].
14. Petrovich, M., Black, M.J., Varol, G. (2023). "TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis." *ICCV 2023*. arXiv:2305.00976. `https://mathis.petrovich.fr/tmr`
15. Tseng, J., Castellon, R., Liu, C.K. (2023). "EDGE: Editable Dance Generation From Music." *CVPR 2023*. arXiv:2211.10658.
16. Xie, J. et al. (2024). "OmniControl: Control Any Joint at Any Time for Human Motion Generation." arXiv:2310.08580.
17. Tevet, G., et al. (2023). "Human Motion Diffusion Model (MDM)." arXiv:2209.14916. *(foundation for editing via inpainting)*

---

## KEY ACTIONABLE TAKEAWAYS FOR AgentLODGE

1. **Use the AgentBanana architecture pattern, adapted for motion:** propose → apply → verify → refine, bounded at MAX_ITER=4.

2. **Implement a bounded MEO vocabulary** (10–15 operations) as in Goel et al. This prevents the LLM from hallucinating impossible operations and makes the loop tractable. Every NL instruction must map to this vocabulary.

3. **TMR is your primary verifier signal.** It's objective, fast (cosine similarity in embedding space), and directly measures text-motion semantic alignment, the core thing the user cares about. Use it as the primary metric, not LLM self-evaluation.

4. **Layer the verification signals:** TMR (semantic) + beat alignment score (musical sync) + energy delta (physical intensity) + optional VLM (Nano Banana on rendered video). This multi-signal approach prevents gaming of any single metric.

5. **Do not rely on pure LLM self-critique** without external signals, Huang et al. (2023) shows it typically hurts performance after iteration 1. The Reflexion memory buffer works when combined with real environment feedback.

6. **Store the Reflexion-style memory buffer** across sessions: "When user asks for 'more energetic', amplitude_scale=1.4 worked for this style; amplitude_scale=1.8 was too extreme → clipped." This enables per-user personalization over time.

7. **EDGE's in-betweening** is the exact tool needed for "keep this part, change that part": fix keyframes at segment boundaries, let diffusion regenerate the interior. Combine with OmniControl's joint-level constraints for "add spin at 0:30."

8. **Partial success is a valid output.** Return the best result achieved after MAX_ITER with a clear explanation of the gap and suggested manual override parameters. Never silently return the original unchanged dance.