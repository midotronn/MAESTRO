Agent is idle (waiting for messages). agent_id: lit-llm-motion-text, agent_type: research, status: idle, description: Research LLM reasoning & motion text, elapsed: 850s, total_turns: 1

[Turn 0]
I now have sufficient data to compile the full literature review. Let me write it up comprehensively.

---

# Comprehensive Literature Review: Text Descriptions for Dance Segment Reasoning in AgentLODGE

**Compiled for:** AgentLODGE, LLM-Storyboard + Training-Free Assembler over LODGE & EDGE dance diffusion generators
**Goal of this review:** Establish the research foundation for using LLM-generated / motion-captioner-generated *text descriptions* of dance segments to strengthen the reasoning and selection capabilities of the storyboard agent.

---

## Executive Summary

The core idea, caption a synthesized dance segment in words, feed those captions to the storyboard LLM for grounded comparison and selection, and then verify the realized segment against the intended plan using a cross-modal similarity score, rests on a mature body of work spanning three literatures: (a) **text↔motion models** that already support bidirectional translation between 3D motion sequences and natural-language descriptions; (b) **LLM reasoning methods** (CoT, self-consistency, ReAct, Reflexion) that consistently show that *intermediate natural-language representations* improve decision quality on long-horizon tasks; and (c) **retrieval-augmented and verifier-augmented generation** frameworks that demonstrate the value of grounding LLM reasoning in concrete, task-relevant descriptions. Dance-specific captioning is still nascent: no production-quality model has been explicitly trained on choreography-vocabulary text, but both MotionGPT and MotionLLM generalize usefully to dance-like motions, and the gap can be partially closed by kinematic-feature-based captions derived from velocity, energy, foot contacts, periodicity, and spatial extent. TMR (Petrovich et al., 2023) provides an immediately usable, open-source cross-modal similarity scorer that can serve as the alignment-verification critic in the pipeline.

---

## Section 1: Text ↔ Motion Models

### 1.1 Motion Captioning (Motion → Text)

#### TM2T, Stochastic and Tokenized Modeling for Reciprocal Generation

**Citation:** Guo et al., "TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts," *ECCV 2022*. arXiv:2207.01696. [[link](https://arxiv.org/abs/2207.01696)]

**Key contribution:** TM2T is the *first* model explicitly designed for **both** text-to-motion and **motion-to-text** (i.e., motion captioning) as first-class tasks. Motion sequences are tokenized into a discrete vocabulary via a VQ-VAE; an NMT-style auto-regressive transformer then translates between motion token sequences and text tokens. The motion2text module is trained with an *inverse alignment loss* that penalizes large deviation of predicted captions from ground-truth text, which empirically prevents degenerate outputs (e.g., trivial motionless poses).

**Relevance to AgentLODGE:** TM2T is the go-to lightweight motion captioner. Given a LODGE or EDGE-generated segment (as a joint-angle sequence or SMPL parameters), TM2T can produce a free-text description. The model is publicly available at the project page (https://ericguo5513.github.io/TM2T/). It is trained on HumanML3D and KIT-ML datasets, so captions are constrained to the vocabulary of those annotations (locomotion and simple action verbs), but provide a useful baseline for non-dance-specific captioning.

**Actionable takeaway:** Use TM2T as the *baseline* captioner for dance segments. Its stochastic decoding (NMT with temperature sampling) means multiple candidate captions can be sampled and the most consensus-like one selected (connecting to self-consistency, §2.1).

---

#### MotionGPT, Human Motion as a Foreign Language

**Citation:** Jiang et al., "MotionGPT: Human Motion as a Foreign Language," *NeurIPS 2023*. arXiv:2306.14795. [[link](https://arxiv.org/abs/2306.14795)]

**Key contribution:** MotionGPT unifies motion-language tasks by treating human motion as a "foreign language." 3D motion is first discretized via VQ-VAE into motion tokens, which are added to the vocabulary of a pre-trained LLM (T5-based). Motion captioning, text-to-motion, motion prediction, and motion in-between are all handled as seq2seq tasks on this unified token sequence. The model is instruction-tuned using a curated prompt set for each task.

**Relevance to AgentLODGE:** MotionGPT directly supports **motion captioning** as an instruction-following task. You can query it with a prompt like: *"Describe the following motion sequence: [motion tokens]"* and receive a natural-language description. Because it is built on an LLM backbone, it can produce more fluent, semantically rich descriptions than TM2T. Importantly, MotionGPT's motion captioning quality was demonstrated to be state-of-the-art on HumanML3D.

**Actionable takeaway:** MotionGPT is the **preferred learned captioner** for AgentLODGE segments. Its LLM backbone also means it can be prompted to generate descriptions in specific styles (e.g., "describe in terms of energy level, spatial extent, and body part emphasis").

---

#### AvatarGPT, All-in-One Motion Understanding, Planning, Generation

**Citation:** Zhou, Wan, Wang, "AvatarGPT: All-in-One Framework for Motion Understanding, Planning, Generation and Beyond," *CVPR 2024*. arXiv:2311.16468. [[link](https://arxiv.org/abs/2311.16468)]

**Key contribution:** AvatarGPT extends the GPT-for-motion paradigm to include **motion understanding (captioning), planning, and generation** in a single closed-loop framework where language is the universal interface. A key innovation is an *unsupervised pipeline that generates natural-language descriptions of human action sequences from in-the-wild videos*, enabling training data creation without manual annotation. The closed-loop enables iterative long-motion synthesis by repeatedly: caption → plan → generate → caption.

**Relevance to AgentLODGE:** Two components are directly relevant: (1) **motion captioning** that can describe dance segments in natural language; and (2) the *closed-loop architecture* which is conceptually identical to what AgentLODGE proposes, caption a segment, feed the description to a planner, plan the next segment. The unsupervised video-based annotation pipeline could be adapted to create dance-specific training data by processing dance videos labeled with choreography styles.

**Actionable takeaway:** AvatarGPT demonstrates the feasibility of the full AgentLODGE captioning-planning loop and provides a blueprint. Consider using its video-based caption generation approach to create training data for a dance-specific motion captioner.

---

#### MotionLLM, Understanding Human Behaviors from Motions and Videos

**Citation:** Chen, Lu, Zeng, Zhang, Wang, Zhang, Zhang, "MotionLLM: Understanding Human Behaviors from Human Motions and Videos," 2024. arXiv:2405.20340. [[link](https://arxiv.org/abs/2405.20340)]

**Key contribution:** MotionLLM bridges the LLM literature with motion understanding by jointly training on both video and SMPL motion sequences. It achieves strong results on **captioning, spatial-temporal comprehension, and reasoning**. The joint video+motion training lets the model understand fine-grained body part dynamics better than video-only or motion-only models. A new dataset (MoVid) and benchmark (MoVid-Bench) are introduced for evaluation.

**Relevance to AgentLODGE:** MotionLLM can caption a rendered dance segment (video) or a motion file (SMPL) and produce rich descriptions covering spatial-temporal dynamics. Because dance videos are widely available, MotionLLM's video pathway may be particularly useful, you can render a LODGE/EDGE segment as video frames and feed them directly.

**Actionable takeaway:** If rendering segments as video is feasible in the AgentLODGE pipeline, MotionLLM offers a **richer captioning pathway** than motion-only models, with the ability to reason about fine-grained spatial and temporal structure.

---

### 1.2 Text-Motion Alignment/Retrieval (as Critic/Verifier)

#### TMR, Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis

**Citation:** Petrovich, Black, Varol, "TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis," *ICCV 2023*. arXiv:2305.00976. [[link](https://arxiv.org/abs/2305.00976)]

**Key contribution:** TMR extends TEMOS with a *contrastive loss* (CLIP-style) to build a joint text-motion embedding space. The trained encoder maps both text descriptions and 3D motion sequences into a shared latent space where cosine similarity measures semantic alignment. The contrastive loss substantially improves retrieval: median rank drops from 54 to 19 on HumanML3D. Code and models are publicly available at https://mathis.petrovich.fr/tmr.

**Relevance to AgentLODGE, core critic/verifier:** TMR's cross-modal similarity score is directly usable as an **alignment metric**: given a plan-phase text description (the storyboard LLM's intended choreography), compute the TMR cosine similarity between that description and each candidate dance segment. This provides a numerically grounded verification step. The formula is:
```
alignment_score = cosine_sim(TMR_text_encoder(description), TMR_motion_encoder(segment))
```
Segments that achieve high alignment score with the storyboard intent should be preferred.

**Important nuance:** TMR's motion encoder is trained on HumanML3D (general human motion). Dance segments may exhibit out-of-distribution joint-angle patterns. Fine-tuning TMR on a dance-motion/text paired dataset (see AIST++ §1.3) would strengthen its discriminative power for dance.

**Actionable takeaway:** TMR is the **recommended alignment verifier** for AgentLODGE. It is open-source, inference-efficient, and outputs interpretable cosine similarity scores. Use TMR to re-rank candidate LODGE/EDGE segments based on alignment with the storyboard description.

---

#### TEMOS, Generating Diverse Human Motions from Textual Descriptions

**Citation:** Petrovich, Black, Varol, "TEMOS: Generating diverse human motions from textual descriptions," *ECCV 2022*. arXiv:2204.14109. [[link](https://arxiv.org/abs/2204.14109)]

**Key contribution:** TEMOS introduced a VAE-based framework with separate text and motion encoders trained to match distributions in a shared latent space. It demonstrated that a well-trained cross-modal encoder can assess text-motion alignment, which TMR later extended contrastively. The TEMOS encoder itself can serve as a weaker alignment scorer.

---

#### MDM, Human Motion Diffusion Model

**Citation:** Tevet, Raab, Gordon, Shafir, Bermano, Cohen-Or, "Human Motion Diffusion Model," *ICLR 2023*. arXiv:2209.14916. [[link](https://arxiv.org/abs/2209.14916)]

**Key contribution:** MDM is a transformer-based, classifier-free diffusion model for text-conditioned motion generation. A notable design choice is predicting the *sample* (rather than the noise) at each diffusion step, which enables imposing geometric losses (foot contact, velocity) during training. MDM supports multiple conditioning modalities (text, action classes, prefixes) and achieves SOTA on HumanML3D and HumanAct12.

**Relevance to AgentLODGE:** MDM's foot contact loss and velocity loss framework provides a blueprint for the **kinematic quality features** used in kinematic-feature-based captions (§3.3). MDM's classifier-free guidance mechanism also demonstrates how text can be used to steer motion generation, the same text-motion alignment formalism used in TMR underlies MDM.

---

#### MotionCLIP, Exposing Human Motion Generation to CLIP Space

**Citation:** Tevet, Gordon, Hertz, Bermano, Cohen-Or, "MotionCLIP: Exposing Human Motion Generation to CLIP Space," *ECCV 2022*. arXiv submitted March 2022. [[search result](https://arxiv.org/search/?searchtype=all&query=MotionCLIP+motion+clip+tevet)]

**Key contribution:** MotionCLIP trains a transformer-based motion auto-encoder to align its latent space with CLIP's joint image-text space. This implicitly transfers CLIP's semantic knowledge (trained on 400M image-text pairs) into the motion domain, enabling: (a) text-to-motion generation via CLIP text embeddings without paired motion-text data; (b) out-of-domain generalization (e.g., "Spiderman" → web-swinging motion); (c) disentangled motion editing.

**Relevance to AgentLODGE:** MotionCLIP offers a computationally cheap **alignment proxy**: encode the storyboard description with CLIP's text encoder and score each dance segment by similarity to that encoding in the shared CLIP-aligned motion latent space. Because CLIP is already widely deployed, this requires no motion-specific training on dance data, though the alignment quality will be noisier than TMR.

---

#### T2M-GPT, Generating Human Motion from Textual Descriptions with Discrete Representations

**Citation:** Zhang et al., "T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations," *CVPR 2023*. arXiv:2301.06052. [[link](https://arxiv.org/abs/2301.06052)]

**Key contribution:** T2M-GPT combines VQ-VAE motion tokenization with GPT-style auto-regressive generation conditioned on CLIP text embeddings. Its VQ-VAE provides a discrete motion vocabulary that can be used for *motion-to-text* retrieval by nearest-neighbor lookup in token space.

---

#### MoMask, Generative Masked Modeling of 3D Human Motions

**Citation:** Guo et al., "MoMask: Generative Masked Modeling of 3D Human Motions," *2023*. arXiv:2312.00063. [[link](https://arxiv.org/abs/2312.00063)]

**Key contribution:** MoMask applies BERT-style masked modeling to motion token sequences, enabling bidirectional generation and editing. It achieves strong text-conditioned generation with the ability to complete or edit partial motion sequences, highly relevant for the segment-stitching step in AgentLODGE's assembler.

**Relevance to AgentLODGE:** MoMask's bidirectional modeling could support *masked in-betweening*, given a textual intent for a segment, generate only the missing frames to stitch two segments together while respecting both the text description and the boundary constraints.

---

### 1.3 Dance-Specific Description and Annotation

#### AIST++, Music Conditioned 3D Dance Generation with AIST++

**Citation:** Li et al., "AI Choreographer: Music Conditioned 3D Dance Generation with AIST++," *ICCV 2021*. arXiv:2101.08779. [[link](https://arxiv.org/abs/2101.08779)]

**Key contribution:** AIST++ provides 5.2 hours of 3D dance motion in 1408 sequences across **10 dance genres** (breakdance, waacking, locking, popping, etc.) with multi-view camera poses and music. The genre labels (e.g., "breakdance," "waacking") serve as coarse, structured text descriptors of dance style, a form of dance-specific annotation.

**Relevance to AgentLODGE:** (a) AIST++ genre labels can be used as coarse dance-style tags in storyboard plans; (b) the dataset can be used to fine-tune motion captioning models (TM2T, MotionGPT) on dance-specific vocabulary; (c) AIST++ motions paired with genre labels provide training data for a dance-specific TMR fine-tuning. The FACT model (Full-Attention Cross-modal Transformer) introduced alongside AIST++ sets a baseline for music-conditioned generation quality.

---

#### Dance-Specific Text Description: Gap in the Literature

**Finding:** No published model to date has been specifically trained to produce **choreography-vocabulary text descriptions** (mentioning Laban Movement Analysis features, specific footwork patterns, arm port de bras, formation shapes, musicality, style markers like "isolations" or "undulations"). The closest works are:

1. **AvatarGPT** (arXiv:2311.16468): can be prompted to describe dance from videos, but is not fine-tuned on dance vocabulary
2. **MotionLLM** (arXiv:2405.20340): multi-modal video+motion understanding, closest to describing dance from video
3. **AIST++ genre labels**: coarse but dance-specific

**Actionable recommendation:** To create dance-specific captions, use a two-stage approach: (a) run MotionGPT or TM2T for motion-level captioning; (b) pass that caption + computed kinematic features (energy, periodicity, foot contact rhythm) to a general-purpose LLM (GPT-4/Claude) with a choreography-vocabulary prompt template. This hybrid approach can approximate dance-specific description without requiring new training data.

---

## Section 2: LLM Planning and Reasoning via Intermediate Text

### 2.1 Chain-of-Thought, Plan-and-Solve, Self-Consistency

#### Chain-of-Thought Prompting (CoT)

**Citation:** Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," *NeurIPS 2022*. arXiv:2201.11903. [[link](https://arxiv.org/abs/2201.11903)]

**Key contribution:** CoT shows that prompting an LLM to produce intermediate reasoning steps ("let's think step by step") before giving a final answer substantially improves performance on complex, multi-step tasks. The intermediate steps act as a *natural-language scratchpad* that decompose the problem into tractable sub-problems.

**Relevance to AgentLODGE:** The storyboard LLM can be prompted to reason step by step when selecting a dance segment: *"First, describe what this segment looks like based on its caption. Then, evaluate whether it matches the intended choreography theme. Finally, decide if it should be selected."* This mirrors CoT and is directly actionable.

---

#### Self-Consistency Improves Chain-of-Thought Reasoning

**Citation:** Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," *ICLR 2023*. arXiv:2203.11171. [[link](https://arxiv.org/abs/2203.11171)]

**Key contribution:** Self-consistency samples multiple reasoning paths from the LLM and takes a majority vote over final answers. This is more reliable than greedy decoding, especially for tasks with ambiguous intermediate steps. Applied to CoT, it achieves state-of-the-art on arithmetic and commonsense reasoning benchmarks.

**Relevance to AgentLODGE:** When the storyboard LLM must select among multiple candidate segments, sample 3–5 independent selection reasoning chains and take a vote. This is especially valuable when descriptions are ambiguous or the alignment scores are close. The same principle applies to the storyboard planning phase: generate multiple candidate plans and select by consistency.

---

#### Plan-and-Solve Prompting

**Citation:** Wang et al., "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models," *ACL 2023*. arXiv:2305.04091. [[link](https://arxiv.org/abs/2305.04091)]

**Key contribution:** Plan-and-Solve extends zero-shot CoT by explicitly adding a *planning phase*: "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step by step." This two-phase approach (plan → execute) substantially improves zero-shot performance across diverse reasoning benchmarks.

**Relevance to AgentLODGE:** The storyboard agent is literally a planner. Plan-and-Solve prompting means the LLM should first articulate a *high-level choreographic plan* (section structure, mood trajectory, genre transitions), then execute it segment by segment. Each execution step benefits from having the text description of the realized segment to assess whether the plan is being followed.

---

### 2.2 Reflexion, ReAct, and Verbal Feedback Loops

#### Reflexion, Language Agents with Verbal Reinforcement Learning

**Citation:** Shinn, Cassano, Gopalan, Narasimhan, Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning," *NeurIPS 2023*. arXiv:2303.11366. [[link](https://arxiv.org/abs/2303.11366)]

**Key contribution:** Reflexion proposes that LLM agents can improve decision quality through *verbal reflection* rather than weight updates. After each action, the agent verbally reflects on the outcome (using natural language feedback from the environment or a self-evaluation), stores the reflection in an episodic memory, and uses it to improve subsequent decisions. Reflexion achieves 91% pass@1 on HumanEval, surpassing GPT-4.

**Relevance to AgentLODGE:** Reflexion provides the theoretical basis for the **description-based verification loop** in AgentLODGE. After the assembler stitches a segment, the TMR alignment score (and the segment's text description) can be fed back to the storyboard LLM as *verbal feedback*: *"The segment was described as 'energetic breakdance with fast footwork' but the alignment score to the intended 'smooth lyrical flow' was only 0.31. Reflect on this and revise the next selection."* This reflexive loop would constitute a verbal reinforcement signal without any weight updates.

**Actionable takeaway:** Implement Reflexion-style verbal feedback in AgentLODGE: segment description → alignment check → verbal reflection prompt → revised selection for next segment. This is a training-free improvement to the base storyboard agent.

---

#### ReAct, Synergizing Reasoning and Acting in Language Models

**Citation:** Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," *ICLR 2023*. arXiv:2210.03629. [[link](https://arxiv.org/abs/2210.03629)]

**Key contribution:** ReAct interleaves *reasoning traces* (chain-of-thought style internal monologue) with *actions* (tool calls to external environments). The reasoning traces help the agent plan what to do; the actions execute those plans and return observations; subsequent reasoning uses those observations. This produces more grounded, factual, and coherent agent behavior than either reasoning or acting alone.

**Relevance to AgentLODGE:** ReAct is an excellent paradigm for the storyboard agent's decision loop: the LLM *reasons* about which segment to select next (using descriptions and alignment scores as observations), *acts* by querying the assembler for candidate segments, then *reasons* again about what was returned. The dance segment descriptions are exactly the "observations" that ground the reasoning traces.

---

#### HuggingGPT / TaskMatrix, Orchestration of Specialist AI Models via LLM

**Citation:** Shen et al., "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face," *ICLR 2024*. arXiv:2303.17580. [[link](https://arxiv.org/abs/2303.17580)]

**Key contribution:** HuggingGPT uses ChatGPT as a *controller* that receives a user request, plans which specialist AI models to invoke (from HuggingFace), parses their outputs, and generates a final response. The key insight is that an LLM can orchestrate heterogeneous sub-models by describing tasks and interpreting results in natural language.

**Relevance to AgentLODGE:** AgentLODGE is structurally identical to HuggingGPT: the storyboard LLM orchestrates LODGE, EDGE, and the TMR verifier as specialist sub-models. HuggingGPT's planning → execution → response paradigm maps directly to storyboard → assembler → verification. The description interface between the LLM and sub-models should be natural language (just as in HuggingGPT), not raw tensors.

---

### 2.3 LLM as Long-Horizon Planner for Creative Tasks

#### Voyager, An Open-Ended Embodied Agent with Large Language Models

**Citation:** Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," *2023*. arXiv:2305.16291. [[link](https://arxiv.org/abs/2305.16291)]

**Key contribution:** Voyager is an LLM-powered agent that continuously explores Minecraft by: (1) formulating goals, (2) generating code/skills to achieve them, (3) executing them, (4) verifying success via environment feedback, and (5) storing successful skills in a *curriculum library*. It achieves dramatically better exploration than baselines, largely because natural-language skill descriptions make the agent's knowledge compositional and reusable.

**Relevance to AgentLODGE:** Voyager demonstrates that **storing successful plans (as text) in a library** enables compositional long-horizon planning. For AgentLODGE: a library of successful choreography segment descriptions (indexed by dance style, energy, music feature) could be maintained and retrieved to seed planning for new pieces. This connects to RAG-style retrieval (§3.1).

---

#### Reflexion and Creative-Task LLM Planners

**Research gap:** We found no published paper that specifically applies CoT/Plan-and-Solve/Reflexion to music-to-dance choreography planning. The closest are:

- **CHOREOMASTER** (Chen et al., SIGGRAPH 2021): a graph-based choreography planning system that selects motion clips from a database using a music rhythm graph, but does **not** use LLMs or natural-language descriptions.
- **AvatarGPT** (arXiv:2311.16468): demonstrates LLM-based motion planning via the language interface, including long-motion synthesis via iterative closed-loop planning.

**Actionable takeaway:** AgentLODGE would be the *first* work to apply LLM-based plan-and-solve reasoning (with natural-language segment descriptions as the intermediate representation) to music-to-dance choreography. This is a strong novelty claim.

---

## Section 3: Segment Descriptions for Improved Selection/Reasoning

### 3.1 Retrieval-Augmented Generation (RAG)

**Citation:** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS 2020*. arXiv:2005.11401. [[link](https://arxiv.org/abs/2005.11401)]

**Key contribution:** RAG augments an LLM's input with *retrieved relevant documents* (encoded by a dense retriever and fetched from an external knowledge store). The retrieved content grounds the LLM's generation in verified, task-relevant facts. RAG achieves state-of-the-art on open-domain QA benchmarks.

**Relevance to AgentLODGE:** The segment description library (text captions of all candidate LODGE/EDGE segments) can serve as a RAG knowledge store. When planning the next segment, the storyboard LLM can **retrieve the most relevant past successful segments** by querying the description library with a text query (e.g., "energetic popping with music beat 128 BPM"), then use the retrieved descriptions to condition its plan.

---

### 3.2 Evidence That Richer Descriptions Improve LLM Selection Quality

**Theoretical basis from CoT/ReAct literature:**
- Wei et al. (2022) showed that intermediate text steps reduce errors in multi-step reasoning. Giving the LLM a description like "Segment A: moderate energy, hip-heavy isolations, syncopated rhythm, 0.71 TMR score; Segment B: high energy, full-body jumps, on-beat, 0.43 TMR score" lets it reason compositionally rather than choosing blindly.
- Wang et al. (2023, self-consistency) showed that sampling multiple reasoning chains and taking a vote is more reliable, description diversity (different captions for the same segment) can serve as natural chain diversity.
- Reflexion (Shinn et al. 2023) demonstrates verbal feedback drives iterative improvement, text descriptions of failed/successful segments enable explicit learning without fine-tuning.

**Specific support from motion domain:**
- AvatarGPT (arXiv:2311.16468) explicitly uses language as the universal interface between planning and generation: the planner describes what motion is needed, the generator produces it, the captioner describes the output, and the planner assesses the gap. This *describe-then-assess* loop is precisely what AgentLODGE proposes.
- TMR (arXiv:2305.00976) validates that cross-modal text-motion similarity is a meaningful quality signal: a motion that scores 0.8 cosine similarity against its own ground-truth description is objectively better aligned with that description than one scoring 0.3.

**Broader LLM reasoning literature:**
- The general finding across Chain-of-Thought, Plan-and-Solve, ReAct, and Reflexion is that *richer, more grounded intermediate representations* improve decision quality. Raw numbers (e.g., kinematic energy = 2.3 J/s) are harder for LLMs to reason about than semantic descriptions ("fast-paced, high-energy movement with frequent jumps"). Converting numerical features to language bridges this gap.

---

### 3.3 Kinematic Feature-Based Motion Captioning

**Research basis:** Several dance/motion evaluation works extract kinematic features as a proxy for motion quality and style. The relevant features and their computation methods are:

| Feature | Computation | Dance Meaning |
|---|---|---|
| **Mean kinetic energy** | Mean of squared joint velocities | Overall energy/intensity level |
| **Acceleration variance** | Variance of joint acceleration | Sharpness vs. smoothness |
| **Foot contact ratio** | Fraction of frames with foot near ground | Grounded vs. aerial |
| **Beat alignment score** | Cross-correlation of kinetic energy peaks with music beat times | Music-movement synchrony |
| **Periodicity** | FFT of joint velocities, dominant frequency | Repeated/looped vs. improvised feel |
| **Spatial extent** | Convex hull volume of all joint positions | Full-body vs. compact style |
| **Symmetry** | Mirror symmetry of left/right joint trajectories | Bilateral symmetry (ballet) vs. asymmetric (hip-hop) |
| **Vertical displacement** | Mean height of center-of-mass above floor | Low-to-ground vs. upright |

**Grounding in existing works:**
- **EDGE** (arXiv:2211.10658) introduces a physical plausibility metric based on foot contact scores, directly usable as a captioning feature: *"physical grounding: high/medium/low"*
- **MDM** (arXiv:2209.14916) uses foot contact loss and velocity loss during training, establishing these as canonical quality features
- **AIST++** (arXiv:2101.08779) evaluates beat alignment score (BAS) as a primary dance quality metric: BAS measures the fraction of dance beats that align with music beats within a window

**Actionable template for kinematic-feature-based caption:**
```
"[Energy: {high|medium|low}] [Tempo: {fast|moderate|slow}] 
[Style: {grounded|aerial|mixed}] [Body-focus: {lower-body|upper-body|full-body}]
[Symmetry: {bilateral|asymmetric}] [Music-sync: {on-beat|syncopated|free}]
[Spatial: {expansive|compact|medium}]"
```
This structured caption template is computable from joint kinematics without any learned model, making it computationally cheap (~milliseconds per segment) and deterministic.

---

## Section 4: Actionable Design for AgentLODGE

### 4.1 Step-by-Step Caption Production for Each Candidate Segment

**Stage 1: Kinematic Feature Extraction** (deterministic, fast, ~2ms per segment)

Given a candidate segment (N frames × J joints × 3D positions from LODGE or EDGE):
1. Compute `kinetic_energy(t) = Σ_j ||v_j(t)||²` for each frame
2. Compute `foot_contact(t)` from ankle height + velocity threshold (as in MDM/EDGE)
3. Compute beat alignment: cross-correlate `kinetic_energy` with music beat onsets
4. Compute periodicity from FFT of mean joint velocity norm
5. Compute spatial extent from convex hull of all joint positions across the segment
6. Map all to discrete labels using tuned thresholds and produce a **kinematic tag string**:
   `"energy=high, tempo=fast, foot=aerial, body=full, sync=on-beat, spatial=expansive"`

**Stage 2: Learned Caption** (probabilistic, richer, ~200ms per segment)

Run the kinematic tag string + the raw motion tokens through **MotionGPT** (arXiv:2306.14795) with the prompt:
> *"Given this motion sequence with the following kinematic properties: {kinematic tags}, generate a natural language description of the dance movement in the vocabulary of choreography. Focus on energy, quality of movement, body part usage, and rhythm."*

Sample 3–5 descriptions and use the most consensus-like (self-consistency vote by embedding similarity).

**Output:** Each candidate segment gets a structured description like:
> *"A high-energy breakdance segment with explosive footwork and frequent aerial jumps, strongly aligned to a 128 BPM beat. The movement is predominantly lower-body-driven with syncopated rhythmic accents and wide spatial traversal."*

**Research justification:**
- MotionGPT (arXiv:2306.14795) demonstrates SOTA motion captioning quality on HumanML3D
- AvatarGPT (arXiv:2311.16468) validates the describe-and-plan closed loop
- Self-consistency (arXiv:2203.11171) justifies sampling multiple captions and selecting by vote

---

### 4.2 LLM Reasoning Over Text Descriptions for Segment Selection

**Prompt structure for the storyboard/selection LLM:**

```
SYSTEM: You are a choreography director planning a {genre} dance to music with the 
following structure: {musical_structure}. You have already generated sections 
{1..k-1} with the following choreographic arc: {arc_so_far}.

USER: You must now select one of the following candidate segments for section {k}:
---
SEGMENT A:
  Kinematic summary: energy=medium, tempo=moderate, foot=grounded, sync=syncopated
  Text description: "A smooth, hip-heavy waacking segment with fluid transitions 
  and off-beat accents. Movement is compact with bilateral arm swings."
  TMR alignment to plan intent ("lyrical bridge"): 0.62

SEGMENT B:
  Kinematic summary: energy=high, tempo=fast, foot=aerial, sync=on-beat
  Text description: "An explosive power move sequence with full-body commitment 
  and precise beat-locking. Wide spatial range."
  TMR alignment to plan intent ("lyrical bridge"): 0.28

Let's think step by step:
1. What does the choreographic arc require at this moment?
2. Which segment best fits based on energy, style, and music sync?
3. Which segment has the highest alignment with the intended lyrical bridge?
4. Select A or B and explain why.
```

**Key design principles grounded in literature:**
- **CoT (arXiv:2201.11903):** "Let's think step by step" forces structured reasoning before selection
- **Plan-and-Solve (arXiv:2305.04091):** The arc summary (plan) is re-stated before execution
- **ReAct (arXiv:2210.03629):** The TMR score is an observation that feeds back into the reasoning
- **Reflexion (arXiv:2303.11366):** If a selected segment's realized alignment score is poor, the LLM receives verbal feedback for the next selection decision

---

### 4.3 TMR-Based Alignment Verification

**Implementation:**

1. At plan time: storyboard LLM writes an **intent description** per section, e.g., *"lyrical bridge with moderate energy and smooth transitions."*
2. At realization time: after the assembler stitches the selected segment, compute:
   ```python
   score = cosine_similarity(
       TMR_text_encoder(intent_description),
       TMR_motion_encoder(realized_segment)
   )
   ```
3. **Threshold-based acceptance:** if `score < τ` (e.g., 0.40), flag the section for regeneration and provide verbal feedback to the storyboard LLM: *"Section {k} realized TMR alignment: {score:.2f} against intent '{intent}'. Please revise the selection or regenerate candidates."*
4. **Re-ranking:** Use TMR scores to re-rank multiple LODGE/EDGE candidate segments before presenting them to the LLM, so the LLM's selection is already pre-filtered to high-quality candidates.

**Research justification:**
- TMR (arXiv:2305.00976): public, open-source cross-modal similarity scorer, achieves median rank 19 on HumanML3D
- Reflexion (arXiv:2303.11366): verbal feedback from environment enables iterative improvement without weight updates
- RAG (arXiv:2005.11401): retrieval of high-TMR-score past segments can seed the generation process

**Fine-tuning recommendation:** Fine-tune TMR on AIST++ genre-text pairs to improve its alignment scoring for dance-specific vocabulary. The AIST++ genre labels + segment descriptors (kinematic tag strings, §4.1) would constitute the text side, and AIST++ motion sequences the motion side. This is a modest fine-tuning effort on an already-trained contrastive model.

---

### 4.4 Full AgentLODGE Description-Augmented Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Music audio + desired genre/mood                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  STORYBOARD LLM (Plan Phase)   │ ← CoT + Plan-and-Solve
          │  Produces: section intents     │
          │  e.g., ["energetic intro",     │
          │          "lyrical bridge",     │
          │          "climax finale"]      │
          └────────────────┬───────────────┘
                           │  intent_i per section
                           ▼
          ┌────────────────────────────────┐
          │  DANCE GENERATORS              │
          │  LODGE (characteristic prims)  │
          │  EDGE  (transformer diffusion) │
          │  → K candidate segments each  │
          └────────────────┬───────────────┘
                           │ K candidate segments
                           ▼
          ┌────────────────────────────────────────────────┐
          │  CAPTIONING STAGE                              │
          │  For each candidate segment:                   │
          │  1. Kinematic tag extraction (~2ms)            │
          │     [energy, tempo, foot, sync, spatial]       │
          │  2. MotionGPT caption (~200ms)                 │
          │     → natural language description             │
          │  3. TMR alignment score vs. intent_i           │
          │     → cosine_sim(text_enc(intent), mot_enc)   │
          └────────────────┬───────────────────────────────┘
                           │ {segment, description, tags, TMR_score} × K
                           ▼
          ┌────────────────────────────────────────────────┐
          │  STORYBOARD LLM (Selection Phase)              │ ← ReAct + CoT
          │  Receives: K description bundles               │
          │  Reasons over descriptions + TMR scores        │
          │  Selects best segment via step-by-step CoT     │
          └────────────────┬───────────────────────────────┘
                           │ selected_segment
                           ▼
          ┌────────────────────────────────────────────────┐
          │  TRAINING-FREE ASSEMBLER                       │
          │  Stitches selected segments                    │
          │  → full dance sequence                         │
          └────────────────┬───────────────────────────────┘
                           │ realized section
                           ▼
          ┌────────────────────────────────────────────────┐
          │  VERIFICATION (Reflexion Loop)                 │
          │  TMR(intent_i, realized_section) → score       │
          │  If score < τ: verbal feedback to LLM          │
          │     → LLM reflects, regenerates or re-selects  │
          └────────────────┬───────────────────────────────┘
                           │ verified sections
                           ▼
                     OUTPUT: Full Dance
```

---

## Section 5: Summary of Gaps and Open Questions

| Gap | Severity | Recommended Action |
|---|---|---|
| No dance-specific motion captioning model exists | Medium | Fine-tune MotionGPT on AIST++ with choreography-vocabulary prompts |
| TMR trained on general motion, not dance | Medium | Fine-tune TMR on AIST++ genre-motion pairs |
| MotionCLIP arXiv ID not confirmed by direct fetch | Low (paper confirmed from search) | Cite as "Tevet et al., ECCV 2022, arXiv March 2022" |
| LODGE arXiv ID not confirmed by direct fetch | Low | Cite as "Li et al., CVPR 2024" (title confirmed from arXiv search) |
| No existing work applies description-augmented LLM reasoning to choreography | High opportunity | This is AgentLODGE's key novel contribution |
| LLM reasoning quality on creative selection (vs. arithmetic) less studied | Medium | Use self-consistency + few-shot examples from AIST++ library |

---

## Verified References

1. **TM2T:** Guo, C., et al. "TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts." *ECCV 2022*. arXiv:2207.01696. https://arxiv.org/abs/2207.01696

2. **MotionGPT:** Jiang, B., et al. "MotionGPT: Human Motion as a Foreign Language." *NeurIPS 2023*. arXiv:2306.14795. https://arxiv.org/abs/2306.14795

3. **TMR:** Petrovich, M., Black, M. J., Varol, G. "TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis." *ICCV 2023*. arXiv:2305.00976. https://arxiv.org/abs/2305.00976 | Code: https://mathis.petrovich.fr/tmr

4. **TEMOS:** Petrovich, M., Black, M. J., Varol, G. "TEMOS: Generating diverse human motions from textual descriptions." *ECCV 2022*. arXiv:2204.14109. https://arxiv.org/abs/2204.14109

5. **MDM:** Tevet, G., Raab, S., Gordon, B., Shafir, Y., Bermano, A. H., Cohen-Or, D. "Human Motion Diffusion Model." *ICLR 2023*. arXiv:2209.14916. https://arxiv.org/abs/2209.14916

6. **MotionCLIP:** Tevet, G., Gordon, B., Hertz, A., Bermano, A. H., Cohen-Or, D. "MotionCLIP: Exposing Human Motion Generation to CLIP Space." *ECCV 2022*. arXiv submitted March 2022. [arXiv search confirms: https://arxiv.org/search/?searchtype=all&query=MotionCLIP+motion+clip+tevet]

7. **T2M-GPT:** Zhang, J., et al. "T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations." *CVPR 2023*. arXiv:2301.06052. https://arxiv.org/abs/2301.06052

8. **MLD:** Chen, X., et al. "Executing your Commands via Motion Diffusion in Latent Space." *CVPR 2023*. arXiv:2212.04048. https://arxiv.org/abs/2212.04048

9. **MoMask:** Guo, C., et al. "MoMask: Generative Masked Modeling of 3D Human Motions." 2023. arXiv:2312.00063. https://arxiv.org/abs/2312.00063

10. **AvatarGPT:** Zhou, Z., Wan, Y., Wang, B. "AvatarGPT: All-in-One Framework for Motion Understanding, Planning, Generation and Beyond." *CVPR 2024*. arXiv:2311.16468. https://arxiv.org/abs/2311.16468

11. **MotionLLM:** Chen, L.-H., Lu, S., Zeng, A., Zhang, H., Wang, B., Zhang, R., Zhang, L. "MotionLLM: Understanding Human Behaviors from Human Motions and Videos." 2024. arXiv:2405.20340. https://arxiv.org/abs/2405.20340

12. **EDGE:** Tseng, J., Castellon, R., Liu, C. K. "EDGE: Editable Dance Generation From Music." *CVPR 2023*. arXiv:2211.10658. https://arxiv.org/abs/2211.10658

13. **AIST++:** Li, R., et al. "AI Choreographer: Music Conditioned 3D Dance Generation with AIST++." *ICCV 2021*. arXiv:2101.08779. https://arxiv.org/abs/2101.08779

14. **LODGE:** Li, R., Zhang, Y., Zhang, H., Guo, J., Zhang, Y., Liu, Y., Li, X. "Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives." *CVPR 2024*. Submitted March 2024. [Confirmed from arXiv search; exact arXiv ID not resolved by direct fetch]

15. **Lodge++:** Li, R., et al. "Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns." 2024. Submitted October 2024. [Confirmed from arXiv search]

16. **Chain-of-Thought:** Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*. arXiv:2201.11903. https://arxiv.org/abs/2201.11903

17. **Self-Consistency:** Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023*. arXiv:2203.11171. https://arxiv.org/abs/2203.11171

18. **Plan-and-Solve:** Wang, L., et al. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models." *ACL 2023*. arXiv:2305.04091. https://arxiv.org/abs/2305.04091

19. **Reflexion:** Shinn, N., Cassano, F., Gopalan, A., Narasimhan, K., Yao, S. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*. arXiv:2303.11366. https://arxiv.org/abs/2303.11366

20. **ReAct:** Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. arXiv:2210.03629. https://arxiv.org/abs/2210.03629

21. **Voyager:** Wang, G., et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." 2023. arXiv:2305.16291. https://arxiv.org/abs/2305.16291

22. **HuggingGPT:** Shen, Y., et al. "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face." *ICLR 2024*. arXiv:2303.17580. https://arxiv.org/abs/2303.17580

23. **RAG:** Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. arXiv:2005.11401. https://arxiv.org/abs/2005.11401

---

## Gaps and Uncertainties

| Item | Status |
|---|---|
| LODGE exact arXiv ID | **Unresolved.** Title, authors, venue (CVPR 2024) confirmed by arXiv search. Direct arXiv ID guesses (2403.10208, 2403.10284, etc.) all returned unrelated papers. Recommend: verify via `https://arxiv.org/search/?searchtype=all&query=lodge+dance+diffusion+primitives` when arXiv rate-limit lifts |
| MotionCLIP exact arXiv ID | **Unresolved by direct fetch.** Tevet et al., March 2022 ECCV submission confirmed by arXiv search. Exact ID not returned because several March 2022 guesses hit unrelated papers |
| TMR++ | **Not found as a distinct publication.** The WaMo paper (2025) is a follow-on work on fine-grained text-motion retrieval. No paper explicitly named "TMR++" was found |
| Dance-specific captioning models | **Not found.** Confirmed gap in literature as of July 2026 |
| AvatarGPT motion captioning benchmark numbers on dance | **Not verified.** Paper reports SOTA on general motion tasks; dance-specific evaluation not found |
| MotionLLM dance-specific evaluation | **Not verified.** Paper focuses on general human motion; dance scenes in MoVid dataset composition unclear |
| Self-consistency applied to creative selection (not reasoning) | **Inferred.** Self-consistency is validated on arithmetic/reasoning; direct experimental validation on creative selection tasks not found |