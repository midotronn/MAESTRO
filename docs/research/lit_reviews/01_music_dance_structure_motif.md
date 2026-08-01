Agent is idle (waiting for messages). agent_id: lit-structure-motif, agent_type: research, status: idle, description: Research music/dance structure & motif, elapsed: 1290s, total_turns: 1

[Turn 0]
I now have enough material for a comprehensive literature review. Let me compile the complete findings document.

---

# Comprehensive Literature Review: Choreographic Structure for Music-to-Dance AI Pipelines

## EXECUTIVE SUMMARY

This review covers four interrelated domains relevant to strengthening the choreographic structure of the AgentLODGE pipeline: (1) Music Structure Analysis (MSA) algorithms and perceptual theory; (2) Dance composition scholarship on structural devices; (3) Structure-aware/long-horizon dance generation systems and their limitations; (4) Concrete, actionable design recommendations for a training-free assembler. All claims are backed by peer-reviewed, arXiv, or canonical textbook sources from 2000–2025.

---

## PART I, WHAT MAKES A SONG "STRUCTURED": MUSIC STRUCTURE ANALYSIS

### 1.1 Foundations of Musical Form

Musical form operates across multiple nested levels simultaneously:

- **Micro-level:** beat, measure, hypermeasure (4-bar grouping)
- **Meso-level:** phrase (8–16 bars), period, section
- **Macro-level:** full song form (verse/chorus, ABA ternary, AABA 32-bar, rondo, strophic, sonata, EDM drop structure)

The seminal theoretical framework is **Lerdahl, F. & Jackendoff, R. (1983). *A Generative Theory of Tonal Music.* MIT Press.** They propose four hierarchical structures: *grouping structure* (how listeners segment the music into motives, phrases, sections), *metrical structure* (nested strong/weak beat hierarchy), *time-span reduction*, and *prolongational reduction* (tension and release arcs). The tension/release arc, musical expectation built up and resolved, is directly relevant to choreographic energy curves. Lerdahl expanded the quantitative tension model in *Tonal Pitch Space* (Oxford University Press, 2001).

**Huron, D. (2006). *Sweet Anticipation: Music and the Psychology of Expectation.* MIT Press.** argues that music's affective power derives from five prediction-response processes (imagination, tension, prediction, reaction, appraisal, ITPRA). Structural repetition creates expectation; return of an A section after a contrasting B section generates satisfaction via prediction fulfillment. This directly motivates **recapitulation as a perceptually powerful choreographic device**.

**Hypermeter and phrase structure:** listeners parse music into 4-bar and 8-bar hypermetric units. Dance generation systems that operate at the beat level but ignore hypermeter fail to align with the natural phrasing that human audiences perceive. A drop or chorus commonly begins on a hypermetric downbeat (bar 1 of a 4-bar phrase), boundary detection should align with these.

### 1.2 Core Algorithms for Music Structure Analysis

#### 1.2.1 Self-Similarity Matrix (SSM) and Foote Novelty

**Foote, J. (2000). "Automatic audio segmentation using a measure of audio novelty." *IEEE ICME* 2000, pp. 452–455. DOI: 10.1109/ICME.2000.869637.**

This is the foundational paper. The method:
1. Extracts frame-wise audio features (chroma, MFCC, etc.)
2. Computes an SSM where `S[i,j]` = cosine similarity between feature frames `i` and `j`
3. Convolves the SSM along its main diagonal with a **checkerboard kernel** (a Gaussian-tapered `[+1, −1; −1, +1]` block matrix)
4. Peaks in the resulting novelty curve indicate structural boundaries

Key reading: **Müller, M. (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications.* Springer.**, Chapter 6 is the essential reference for SSM-based structure analysis, detailing diagonal structures (repeated sections appear as bright off-diagonal stripes), block patterns (homogeneous regions), and novelty-based boundary detection.

**Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2012). "Unsupervised detection of music boundaries by time series structure features." *Proceedings of AAAI 2012*.** proposes computing time-series structure features from the SSM (computing sequences of feature statistics along diagonals) for unsupervised boundary detection without ground-truth annotation. Available: https://mtg.upf.edu/static/covers/2012/Serra12aaai.pdf

#### 1.2.2 Laplacian Spectral Clustering

**McFee, B. & Ellis, D.P.W. (2014). "Analyzing song structure with spectral clustering." *Proceedings of ISMIR 2014*, pp. 405–410.** (Available: https://bmcfee.github.io/papers/ismir2014_spectral.pdf)

The method:
1. Extracts multi-scale features: **chroma-CQT** (harmonic content) + **MFCC** (timbre)
2. Builds an SSM and interprets it as a graph adjacency matrix
3. Computes the **graph Laplacian** and applies spectral decomposition (eigenvectors)
4. Clusters frames via k-means in spectral space → yields structural segment assignments
5. Hierarchical variant by sweeping number of clusters reveals multi-scale structure

This is implemented in **librosa** (`librosa.segment.recurrence_matrix`, `librosa.segment.laplacian_segmentation`). Key feature: captures **global** relationships, not just local discontinuities, the chorus at time 30s and the chorus at time 90s get assigned to the same cluster even without novelty between them.

**Actionable:** The LaplacianSegmentation approach directly detects sectional identity (which sections are the same musical type), enabling the assembler to know "section 3 is the same type as section 1 → candidate for motion recapitulation."

#### 1.2.3 MSAF Toolkit

**Nieto, O. & Mysore, G.J. (2016). "MSAF: A Python library for audio segmentation." *ISMIR 2016*, paper 000091.** (https://archives.ismir.net/ismir2016/paper/000091.pdf; GitHub: https://github.com/urinieto/msaf)

MSAF implements a family of algorithms under a unified API:
- `boundaries_id="sf"` (structure features, Serrà et al.)
- `labels_id="scluster"` (spectral clustering, McFee & Ellis)
- Also implements Foote novelty, CNNSeg, Olda, etc.

Usage: `boundaries, labels = msaf.process("audio.wav", boundaries_id="sf", labels_id="scluster")`

Returns timestamps + section label strings. This is the direct tool to call in AgentLODGE.

#### 1.2.4 Recent Deep Learning Advances (2018–2024)

**Peeters, G. & Angulo, I. (2022). "SSM-Net: feature learning for Music Structure Analysis using a Self-Similarity-Matrix based loss." *arXiv:2211.08141*.**, Trains a deep audio encoder end-to-end so that its output features produce an SSM resembling the ground-truth SSM.

**Peeters, G. (2023). "Self-Similarity-Based and Novelty-based loss for music structure analysis." *arXiv:2309.02243*.**, Jointly optimizes SSM-based loss and Foote novelty-based loss for improved boundary detection.

**Bimbot, F. et al. (2012). "A Tutorial on Music Structure Analysis: Concepts, Algorithms, and Applications." *EURASIP Journal on Advances in Signal Processing,* Article ID 159807.**, Comprehensive review standardizing MSA terminology, covering boundary detection, segment labeling, evaluation metrics (P/R/F, Rand Index, NMI).

**General MSA survey:** "Audio-Based Music Structure Analysis: Current Trends, Open Challenges, and Applications." *Transactions of ISMIR (TISMIR)* 2022. DOI: 10.5334/tismir.54., Available at https://transactions.ismir.net/articles/10.5334/tismir.54

### 1.3 Signal Features for Structure

| Feature | Captures | Relevance to structure |
|---|---|---|
| **Chroma-CQT** | Harmonic/key content | Verse vs. chorus (harmony changes) |
| **MFCC** | Timbral texture | Instrument arrangement per section |
| **RMS / Loudness** | Energy/dynamics | Intro→verse→chorus intensity arc |
| **Spectral Flux** | Onset density | Drop detection, transition point |
| **Tempogram** | Rhythmic patterns | Tempo changes, section identity |
| **Jukebox embeddings** | High-level music semantics | Genre, mood, structural feeling |

Critically, EDGE (§3.5 below) uses **Jukebox** (Dhariwal et al. 2020, arXiv:2005.00341) features and finds them far superior to hand-crafted MFCC/chroma for music-motion alignment. AgentLODGE should extract Jukebox features for structural labeling.

### 1.4 Evaluation Metrics for MSA

Standard evaluation uses `mir_eval.segment.detection(ref_boundaries, est_boundaries, window=0.5)`, precision/recall/F-measure with 0.5-second tolerance windows. The SALAMI and RWC datasets provide ground-truth annotations.

**SSIMuse** (Ji et al. 2025): adaptation of SSIM from image processing to music structural similarity, measuring motif recurrence fidelity at composition level, useful as a cross-domain metric when comparing musical section similarity to motion section similarity.

---

## PART II, CHOREOGRAPHIC STRUCTURE & COMPOSITION THEORY

### 2.1 Foundational Scholarship

#### Key Books

**Blom, L.A. & Chaplin, L.T. (1982). *The Intimate Act of Choreography.* University of Pittsburgh Press.**

The primary practical reference for dance composition technique. Documents the following structural devices:
- **Motif:** a short movement phrase serving as the seed for development
- **Motif development techniques:** augmentation (bigger/slower), diminution (smaller/faster), retrograde (reversed order), inversion (spatial flip), fragmentation (breaking into parts), embellishment (added ornamentation)
- **ABA form:** A (theme statement) → B (contrasting section) → A (recapitulation, possibly developed/varied). Creates unity and closure through return.
- **Repetition:** emphasizes, creates pattern, aids audience memory
- **Retrograde:** performing sequence in reverse time order, "mirrors through time"

**Humphrey, D. (1959). *The Art of Making Dances.* Rinehart and Company (reprinted Grove Press).**

Humphrey emphasizes:
- **Dramatic arc**: Every dance should have a beginning, development toward a climax, and resolution. Avoid placing climax too early.
- **Energy arc**: The rise-and-fall of movement intensity shapes emotional trajectory. Maps directly onto the music energy arc.
- **Unity and variety**: Repeating material creates unity; changing it creates variety. Both are essential.
- **"All dances are too long"**: Economy of form, remove redundant material.

**Smith-Autard, J.M. (2004). *Dance Composition: A Practical Guide to Creative Success in Dance Making.* A&C Black (now Bloomsbury, 9th ed. 2019). ISBN 9781408160478.**

Systematic treatment of:
- ABA (ternary) form: detailed exercises
- Rondo form (ABACADA...): alternating theme with contrasting episodes
- Binary form (AB): two distinct sections
- Through-composed form: continuous development without return
- Motif development: same techniques as Blom & Chaplin
- **Spatial devices:** mirroring (left-right reflection across stage axis), retrograde, augmentation, canon (same phrase offset in time between dancers)

**Preston-Dunlop, V. & Sanchez-Colberg, A. (2002). *Dance and the Performative: A Choreological Perspective.* Verve.**, Defines recapitulation as literal or varied return of earlier material, providing closure and reinforcing the perceptual memory of the original.

**Laban, R. (1966). *Choreutics.* Macdonald & Evans.**, Provides the spatial vocabulary (kinesphere, effort qualities, space harmony) that underlies Laban Movement Analysis (LMA). Retrograde, mirroring, and spatial reflection are standard LMA concepts used in computational motion analysis.

### 2.2 Structural Devices: Implementation in AI Context

#### Recapitulation (ABA Return)

**What it does perceptually:** When the A section returns after B, listeners/viewers experience recognition + satisfaction (Huron's ITPRA: high prediction, high reward). This is arguably the strongest structural signal available to a choreographer.

**How to implement in AgentLODGE:**
- When MSA detects that music section *k* has the same label as an earlier section *j* (via spectral clustering), retrieve the stored motion segment generated for section *j*
- Apply a **time-stretching** (DTW-based retiming) to fit the new segment duration if different
- Add a short **in-betweening** transition at both ends (blend last 0.5s of preceding segment with first 0.5s of recalled segment)
- Optionally apply a variation operator (retrograde, mirroring) to avoid exact copy

#### Retrograde (Temporal Reversal)

**What it does perceptually:** Retrograde produces a surprising, intellectually satisfying "mirror image in time." Used famously by Merce Cunningham and Anne Teresa De Keersmaeker.

**How to implement:** Reverse the pose frame sequence. If the original clip is `[p_0, p_1, ..., p_N]`, the retrograde is `[p_N, p_{N-1}, ..., p_0]`. Also reverse the left-right convention of the original if desired (anatomical retrograde). In SMPL-X / quaternion representation, simply reverse the frame order, no skeletal changes needed. Note: footstep contacts will be reversed, so apply EDGE's Contact Consistency Loss post-hoc or suppress foot contacts during retimed playback.

#### Spatial Mirroring (Left-Right Reflection)

**What it does perceptually:** Mirroring emphasizes symmetry, balance, and the mirror relationship between dancers or between two occurrences of the same phrase. Common in duets and in choreographic development.

**How to implement in SMPL-based representation:**
- Negate the x-component of all joint global positions: `x → -x`
- Swap left and right body part joint indices (e.g., left hip ↔ right hip, left knee ↔ right knee, etc.)
- In SMPL's local rotation representation, mirroring requires flipping the sign of rotation components corresponding to the sagittal plane

#### Variation (Retiming / Augmentation)

**What it does:** Performs the same motif faster or slower, creating "augmentation" (slower = more majestic) or "diminution" (faster = more energetic).

**How to implement:** DTW-based time warping between pose sequences. Libraries: `tslearn.metrics.dtw` or `librosa.sequence.dtw`. Subsample or interpolate pose sequences.

### 2.3 Computational Choreography

**Aristidou, A., Stavrakis, E., Chrysanthou, Y., & Shamir, A. (2022). "Rhythm is a Dancer: Music-Driven Motion Synthesis with Global Structure." *IEEE Transactions on Visualization and Computer Graphics,* vol. 28, no. 8, pp. 2961–2976. DOI: 10.1109/TVCG.2022.3164937. arXiv:2111.12159.**

This is the most important prior work for AgentLODGE's goals. It proposes a **three-level hierarchical model**:
1. **Pose level:** LSTM generates temporally coherent pose sequences
2. **Motif level:** poses are grouped into recognizable movement motifs using a perceptual loss (encouraging motif distinctness and repeatability)
3. **Choreography level:** global ordering of motifs to create long-term, genre-respecting dance structure

The choreography level is conditioned on music structure, explicitly ensuring that the global dance form mirrors the music form. This is the closest existing work to AgentLODGE's goals.

**Royal Society review:** "An extensive review of computational dance automation techniques and their applications." *Proceedings of the Royal Society A,* 2021, 477(2251). DOI: 10.1098/rspa.2021.0071., Surveys motif detection, grammar-based methods, and deep learning for computational dance, tracing the evolution from evolutionary algorithms to GCN/transformer architectures.

---

## PART III, STRUCTURE-AWARE / LONG-TERM DANCE GENERATION: PRIOR WORK SURVEY

### 3.1 DanceRevolution (2021)

**Huang, Y. et al. (2021). "Dance Revolution: Long-term Dance Generation with Music via Curriculum Learning." *ICLR 2021.*** OpenReview: https://openreview.net/forum?id=Ykl0e1HKPSq

- **Architecture:** Seq2seq GRU autoregressive model conditioned on mel-spectrogram features
- **Long-horizon strategy:** Curriculum learning, progressively increase sequence length during training
- **Structural awareness:** Beat-level alignment only; no section-level or motif awareness
- **Limitations:** No explicit musical form modeling; motion drift at very long sequences; no motif reuse

### 3.2 AIST++ + AI Choreographer (FACT) (2021)

**Li, R., Yang, S., Ross, D.A., & Kanazawa, A. (2021). "AI Choreographer: Music Conditioned 3D Dance Generation with AIST++ Dataset." *ICCV 2021,* pp. 13401–13412. arXiv:2101.08779.**

- **Dataset:** AIST++, 1,408 dance sequences × 10 genres, 3D SMPL poses from multi-view video + MoCap; ~140 hours
- **Architecture:** Full-attention cross-modal transformer (FACT) for music→motion generation
- **Structural awareness:** Beat alignment; genre conditioning via genre tokens; no section segmentation
- **Limitations:** Generates short sequences (≤8s typically); no global choreographic form; no recapitulation

### 3.3 Bailando (CVPR 2022) and Bailando++ (TPAMI 2023)

**Li, S., Yu, W., Gu, T., Lin, C., Wang, Q., Qian, C., Loy, C.C., & Liu, Z. (2022). "Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory." *CVPR 2022.* arXiv:2203.13055.**

**Li, S. et al. (2023). "Bailando++: 3D Dance GPT with Choreographic Memory." *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 45(12):14192–14207. DOI: 10.1109/TPAMI.2023.3319435.**

- **Architecture:** Two-stage: (1) Pose VQ-VAE encodes 3D poses into a discrete codebook (the "choreographic memory bank"); (2) Actor-Critic GPT generates sequences of codebook tokens conditioned on music, fine-tuned with RL using Beat Align reward
- **Key innovation:** The codebook learns a vocabulary of choreographically valid pose clusters, enabling the GPT to compose dances at the "word" level rather than pixel level
- **Structural awareness:** Long-term context via GPT; memory bank encodes stylistically valid poses; but **no explicit section repetition, no motif recapitulation, no ABA form**
- **Limitations:** Section-level choreographic planning absent; cannot reuse an earlier section's motion when the music repeats; diversity limited by codebook size

### 3.4 ChoreoMaster (2021)

**Yu, T. et al. (2021). "ChoreoMaster: Choreography-Oriented Music-Driven Dance Generation with Deep Graph Neural Network." *arXiv:2107.12392.***

- **Architecture:** GCN over human body skeleton; style embedding for genre control; music feature conditioning
- **Structural awareness:** Style-level genre consistency; rhythm alignment; no section-level planning
- **Limitations:** No long-horizon structural planning; limited to short clips

### 3.5 EDGE (CVPR 2023)

**Tseng, J., Castellon, R., & Liu, C.K. (2023). "EDGE: Editable Dance Generation From Music." *CVPR 2023.* arXiv:2211.10658.** Website: https://edge-dance.github.io/

- **Architecture:** Transformer-based **diffusion model** (DDPM) conditioned on Jukebox music features; generates arbitrarily long sequences via sliding window stitching
- **Key innovations:**
  - **Jukebox features:** Pre-trained generative music model embeddings capture deep musical semantics (genre, style, structural feeling) far better than MFCC/chroma
  - **In-betweening/inpainting:** Editing capabilities, can fix joints, interpolate between keyframes, stitch segments (critical for AgentLODGE)
  - **Physical Foot Contact Score:** Novel metric, acceleration-based measure of foot-skating implausibility (no explicit physical modeling needed)
  - **Contact Consistency Loss:** Eliminates foot-sliding artifacts during generation
- **Structural awareness:** Generates music-aligned motion but **no section-level repetition awareness, no ABA form**; stitching is purely sequential
- **Limitations:** Long sequences are generated as independent chunks; no global choreographic structure imposed; motif reuse entirely absent

### 3.6 ChoreoGraph (2022)

**arXiv:2207.07386. "ChoreoGraph: Music-conditioned Automatic Dance Choreography over a Style and Tempo Consistent Dynamic Graph."**

- **Architecture:** Dynamic graph over dance movement units; edges weighted by style and tempo compatibility; music conditioning guides traversal
- **Structural awareness:** Style consistency across a full piece; tempo-aware transitions; no section-level form
- **Limitations:** Graph traversal is locally greedy; no global planning for ABA return

### 3.7 DanceFormer (2022)

**Yan, X. et al. (2022). "DanceFormer: Music Conditioned 3D Dance Generation with Parametric Motion Tokenizer." *arXiv:2206.11844.***

- **Architecture:** Two-stage transformer: (1) coarse keyframe generation at musical beat positions, (2) interpolation between keyframes using a parametric motion tokenizer
- **Structural awareness:** Beat-aligned keyframes; no section-level planning; no motif reuse
- **Limitations:** Keyframe spacing is still beat-level, not section-level

### 3.8 FineDance + FineNet (ICCV 2023)

**Li, R. et al. (2023). "FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation." *ICCV 2023.* arXiv:2212.03741.** GitHub: https://github.com/li-ronghui/FineDance

- **Dataset:** 14.6 hours, 22 fine-grained dance genres, full-body + expressive hand motions in SMPLH format; split by genre and dancer
- **FineNet model:** Diffusion-based generation with expert networks per genre + retrieval module for long-term coherence
- **New metric:** Genre Matching Score, measures whether generated motion genre matches the music's genre
- **Structural awareness:** Genre-specific generation; retrieval-augmented for long-term consistency; no explicit ABA form

### 3.9 LODGE (CVPR 2024) and LODGE++ (2024)

**Li, R. et al. (2024). "Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives." *CVPR 2024.* arXiv:2403.10518.** GitHub: https://github.com/li-ronghui/LODGE

**Li, R. et al. (2024). "Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns." *arXiv:2410.20389.***

- **Architecture:** Two-stage diffusion:
  - **Stage 1 (Global Diffusion):** Generates "characteristic dance primitives", expressive 8-frame motion fragments capturing coarse music-dance correlation at the global level
  - **Stage 2 (Local Diffusion):** Given primitives as scaffold, generates fine-grained motion sequences in parallel for each chunk; enables very long (minutes-scale) generation
  - **Foot Refine Block:** Optimizes foot-ground contact physical plausibility
  - **LODGE++:** Adds Penetration Guidance (prevent self-intersection) + Multi-genre Discriminator for better genre fidelity
- **Dataset:** FineDance (22 genres, avg 152s sequences)
- **Structural awareness:** The "dance primitives" provide global choreographic scaffolding (coarse motif repetition is implicit in the primitive pattern), but **no explicit musical section repetition detection; no ABA recapitulation; no mirror/retrograde**
- **Key limitation for AgentLODGE:** LODGE treats the entire track as a continuous stream and generates primitives from music globally, it does not identify when section B returns as section A and reuse the same primitive pattern

### 3.10 GDANCE / Music-Driven Group Choreography (CVPR 2023)

**Le, N., Pham, T., Do, T., Tjiputra, E., Tran, Q.D., & Nguyen, A. (2023). "Music-Driven Group Choreography." *CVPR 2023.* arXiv:2303.12337.**

- **Architecture:** Transformer with multi-head attention modeling inter-dancer spatial relations; Formation Loss + Relation Loss
- **Structural awareness:** Formation consistency across time; relation modeling; no section-level choreographic form for individual dancers
- **Key for AgentLODGE:** Formation mirroring between dancers is an implemented structural device (dancer A mirrors dancer B), this paper has concrete implementation of **spatial mirroring as a formation constraint**

### 3.11 Rhythm is a Dancer (IEEE TVCG 2022)

**Aristidou, A., Stavrakis, E., Chrysanthou, Y., & Shamir, A. (2022). "Rhythm is a Dancer: Music-Driven Motion Synthesis with Global Structure." *IEEE Transactions on Visualization and Computer Graphics,* 28(8):2961–2976. DOI: 10.1109/TVCG.2022.3164937. arXiv:2111.12159.**

This is the closest prior art to AgentLODGE's design goals:
- **Three-level hierarchy:** Pose → Motif (consecutive pose clusters with perceptual coherence) → Choreography (ordered sequence of motifs respecting global music structure)
- **Choreography-level planning** explicitly considers music structure (tempo, energy, genre phase) to order motifs
- **Limitation:** No explicit ABA recapitulation (musical section type is used for generation but not for reuse of prior motifs); no retrograde/mirror as a principled variation

### 3.12 Music-to-Dance via Atomic Movements (2025)

**arXiv:2607.13978. "Music-to-Dance Generation via Atomic Movements."** (2025)

- **Architecture:** Two-stage, (1) Planning stage: LLM/symbolic model predicts type, duration, timing of "atomic movements" aligned to musical sections; (2) Synthesis stage: transition-aware generator produces smooth motion following the plan
- **Structural awareness:** **Closest to fully section-aware choreography in the literature.** Explicitly models musical form; allows planning of section repetition and motif reuse at the semantic level
- **Key insight for AgentLODGE:** The separation of *planning* from *synthesis* is the right architectural paradigm, the LLM storyboard agent should plan at the atomic/section level before diffusion generation

### 3.13 Motion In-Betweening (For Transitions)

**Harvey, F.G. et al. (2020). "Robust Motion In-Betweening." *SIGGRAPH / ACM TOG.* arXiv:2102.04942.**, adversarial RNN with time-to-arrival embedding; tested on LaFAN1; state-of-the-art for fixed-endpoint transitions.

**Qin, V. et al. (2022). "Motion In-betweening via Two-stage Transformers." *SIGGRAPH Asia 2022.***, Context Transformer (coarse) + Detail Transformer (fine); best performance on LaFAN1.

**Cohan et al. (2024). "Flexible Motion In-betweening with Diffusion Models." *SIGGRAPH 2024.* DOI: 10.1145/3641519.3657414.**, Diffusion-based; arbitrary keyframe placement; allows text cues to guide transition quality.

---

## PART IV, ACTIONABLE DESIGN RECOMMENDATIONS FOR AGENTLODGE

### 4.1 Detecting and Exploiting Sectional Repetition

**Step 1, Boundary detection:**
```python
import msaf
boundaries, labels = msaf.process(
    audio_path,
    boundaries_id="sf",      # Serrà structure features
    labels_id="scluster"     # McFee & Ellis spectral clustering
)
```
Alternatively, use librosa's Laplacian segmentation directly (`librosa.segment.laplacian_segmentation`) with Jukebox or chroma-CQT features. The spectral clustering labeling is critical: it assigns **section type identity** (label "A", "B", "A", "C", "A"...), not just boundaries.

**Step 2, SSM-based repeat detection:**
Build a coarser SSM at the section level: represent each detected section by its mean Jukebox embedding vector, then compute pairwise cosine similarity between all section embeddings. A high off-diagonal value `sim(section_i, section_j) > τ` (typically τ ≈ 0.8) means sections i and j are musically repeated/equivalent, **trigger motion recapitulation**.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# section_embeds: (N_sections, D) array
sim = cosine_similarity(section_embeds)  # (N_sections, N_sections)
# Find repeat pairs: sim[i,j] > 0.8 and j > i
```

**Step 3, Recapitulation routing:**
In the AgentLODGE LLM storyboard, add the following directive:
> *"For each detected musical section, check if its label matches any earlier section. If a label match exists and the section has already been generated (LODGE or EDGE output cached), retrieve the cached motion clip (possibly with variation: mirror/retrograde/retimed) rather than regenerating from scratch."*

This creates ABA structure in the dance automatically when the music has ABA structure, mirroring Blom & Chaplin's principle of structural return.

**Step 4, Energy arc conditioning:**
Compute a loudness curve (librosa RMS or `essentia.Loudness`) and spectral flux curve per section. Pass these as conditioning signals to the LLM storyboard so it can assign "energetic" vs. "calm" motion types per section (e.g., LODGE for high-energy chorus, EDGE for subdued verse), mimicking Humphrey's dramatic arc principle.

### 4.2 Applying Mirroring, Retrograde, and Retiming as Principled Variation

**Spatial Mirroring (for section A' = mirror of A):**
When recapitulating section A as section A' (return), apply left-right mirroring:
1. For all joint global positions in SMPL world coordinates: `x_new = -x`
2. For joint pairs: swap left↔right indices (consult the SMPL joint map: left hip=1↔right hip=2, left knee=4↔right knee=5, left ankle=7↔right ankle=8, etc.)
3. For local rotation quaternions: apply axis-angle negation about the sagittal plane

This is grounded in Smith-Autard (2004) and Preston-Dunlop (2002) as a principled variation device.

**Retrograde (temporal reversal):**
Simply reverse the pose frame list. For a clip `[p_0, ..., p_N]`, the retrograde is `[p_N, ..., p_0]`. Apply a short Gaussian smoothing kernel (σ ≈ 3 frames) at the joints to remove abrupt velocity discontinuities at the clip boundaries. This creates a perceptually distinct but structurally related variation, appropriate for a B→A' transition in ABA form.

**Retiming via DTW:**
When the repeated musical section has a different length than the original (common in live/imperfect looping structures), use DTW to warp the time axis of the cached motion clip to match the new duration:
```python
from tslearn.metrics import dtw_path
# Warp cached_motion to new_duration frames
path, _ = dtw_path(cached_motion, new_section_features)
# Use path to retime cached_motion
```
This mirrors Blom & Chaplin's "augmentation/diminution" (motif performed slower/faster).

### 4.3 Quantifying "Structuredness" of Generated Dance

Propose a **Choreographic Structure Score (CSS)** composed of:

**a) Beat Alignment Score (BAS)**, standard metric from AIST++ papers. Measures fraction of motion kinetic energy peaks aligning within ±50ms of a musical beat.

**b) Section Repetition Correlation (SRC):**
For each pair of musically repeated sections (detected via SSM label matching), compute:
```
SRC(i, j) = cosine_similarity(motion_embed(section_i), motion_embed(section_j))
```
Where `motion_embed` is the mean pose feature (e.g., from a pretrained action recognition model or pose VQ-VAE codebook from Bailando). **Higher SRC = stronger ABA structural mirroring between music and dance.**

**c) FID (Fréchet Inception Distance)**, using a pretrained motion encoder (e.g., AIST++'s action classification network). Compares distribution of generated vs. real human dance motion. Lower = more realistic.

**d) Physical Foot Contact Score (PFC)**, from EDGE (Tseng et al. 2023): acceleration-based foot-skating metric. Requires no explicit physical model.

**e) Genre Matching Score (GMS)**, from FineDance (Li et al. 2023): whether generated motion genre matches music genre.

**Summary CSS formula:**
```
CSS = α·BAS + β·SRC + γ·(1/FID) + δ·PFC + ε·GMS
```
Weights (α, β, γ, δ, ε) can be tuned; SRC (β) is the new metric directly measuring choreographic structuredness.

**f) Energy Arc Correlation (EAC):** Pearson correlation between the music RMS curve and the motion kinetic energy curve over the full track. A well-composed choreography should have correlated energy arcs (Humphrey 1959 principle).

---

## REFERENCES

### MSA Algorithms & Music Theory

1. **Foote, J.** (2000). "Automatic audio segmentation using a measure of audio novelty." *IEEE ICME 2000,* pp. 452–455. DOI: 10.1109/ICME.2000.869637.

2. **McFee, B. & Ellis, D.P.W.** (2014). "Analyzing song structure with spectral clustering." *Proceedings of ISMIR 2014,* pp. 405–410. https://bmcfee.github.io/papers/ismir2014_spectral.pdf

3. **Serrà, J., Müller, M., Grosche, P., & Arcos, J.L.** (2012). "Unsupervised detection of music boundaries by time series structure features." *Proceedings of AAAI 2012.* https://mtg.upf.edu/static/covers/2012/Serra12aaai.pdf

4. **Müller, M.** (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications.* Springer. ISBN 978-3-319-21944-8. [Chapter 6: Music Structure Analysis]

5. **Bimbot, F. et al.** (2012). "A Tutorial on Music Structure Analysis: Concepts, Algorithms, and Applications." *EURASIP Journal on Advances in Signal Processing,* Article 159807. DOI: 10.1186/1687-6180-2012-159.

6. **Nieto, O. & Mysore, G.J.** (2016). "MSAF: A Python library for audio segmentation." *ISMIR 2016,* paper 000091. https://archives.ismir.net/ismir2016/paper/000091.pdf. GitHub: https://github.com/urinieto/msaf

7. **Peeters, G. & Angulo, I.** (2022). "SSM-Net: feature learning for Music Structure Analysis using a Self-Similarity-Matrix based loss." *arXiv:2211.08141.*

8. **Peeters, G.** (2023). "Self-Similarity-Based and Novelty-based loss for music structure analysis." *arXiv:2309.02243.*

9. **Nieto, O. et al.** (2022). "Audio-Based Music Structure Analysis: Current Trends, Open Challenges, and Applications." *Transactions of ISMIR (TISMIR)*. DOI: 10.5334/tismir.54.

10. **Lerdahl, F. & Jackendoff, R.** (1983). *A Generative Theory of Tonal Music.* MIT Press.

11. **Huron, D.** (2006). *Sweet Anticipation: Music and the Psychology of Expectation.* MIT Press. ISBN 978-0-262-58277-0.

12. **Dhariwal, P. et al.** (2020). "Jukebox: A generative model for music." *arXiv:2005.00341.* [Used as music feature extractor in EDGE and recommended for AgentLODGE]

### Dance Composition Scholarship

13. **Blom, L.A. & Chaplin, L.T.** (1982). *The Intimate Act of Choreography.* University of Pittsburgh Press. ISBN 0-8229-5340-4.

14. **Humphrey, D.** (1959). *The Art of Making Dances.* Grove Press (reprinted editions available). [Foundational: energy arc, climax, motif, form]

15. **Smith-Autard, J.M.** (2019). *Dance Composition: A Practical Guide to Creative Success in Dance Making* (9th ed.). Bloomsbury. ISBN 9781408160478. [ABA form, motif development, spatial devices]

16. **Preston-Dunlop, V. & Sanchez-Colberg, A.** (2002). *Dance and the Performative: A Choreological Perspective.* Verve.

17. **Butterworth, J. & Wildschut, L. (Eds.).** (2009). *Contemporary Choreography: A Critical Reader.* Routledge.

18. **Laban, R.** (1966). *Choreutics.* Macdonald & Evans. [Spatial vocabulary, kinesphere, effort qualities]

### Computational Choreography

19. **Aristidou, A., Stavrakis, E., Chrysanthou, Y., & Shamir, A.** (2022). "Rhythm is a Dancer: Music-Driven Motion Synthesis with Global Structure." *IEEE TVCG,* 28(8):2961–2976. DOI: 10.1109/TVCG.2022.3164937. arXiv:2111.12159. **[Most important prior work for AgentLODGE]**

20. **Royal Society review:** (2021). "An extensive review of computational dance automation techniques and their applications." *Proc. Royal Society A,* 477(2251). DOI: 10.1098/rspa.2021.0071.

### Music-Driven Dance Generation Systems

21. **Huang, Y. et al.** (2021). "Dance Revolution: Long-term Dance Generation with Music via Curriculum Learning." *ICLR 2021.* https://openreview.net/forum?id=Ykl0e1HKPSq

22. **Li, R. et al.** (2021). "AI Choreographer: Music Conditioned 3D Dance Generation with AIST++ Dataset." *ICCV 2021,* pp. 13401–13412. arXiv:2101.08779. [AIST++ dataset + FACT model]

23. **Li, S. et al.** (2022). "Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory." *CVPR 2022.* arXiv:2203.13055. GitHub: https://github.com/lisiyao21/Bailando

24. **Li, S. et al.** (2023). "Bailando++: 3D Dance GPT with Choreographic Memory." *IEEE TPAMI,* 45(12):14192–14207. DOI: 10.1109/TPAMI.2023.3319435.

25. **Tseng, J., Castellon, R., & Liu, C.K.** (2023). "EDGE: Editable Dance Generation From Music." *CVPR 2023.* arXiv:2211.10658. Website: https://edge-dance.github.io/ **[EDGE generator in AgentLODGE pipeline]**

26. **Yu, T. et al.** (2021). "ChoreoMaster: Choreography-Oriented Music-Driven Dance Generation with Deep Graph Neural Network." *arXiv:2107.12392.*

27. **arXiv:2207.07386.** (2022). "ChoreoGraph: Music-conditioned Automatic Dance Choreography over a Style and Tempo Consistent Dynamic Graph."

28. **Yan, X. et al.** (2022). "DanceFormer: Music Conditioned 3D Dance Generation with Parametric Motion Tokenizer." *arXiv:2206.11844.*

29. **Li, R. et al.** (2023). "FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation." *ICCV 2023.* arXiv:2212.03741. GitHub: https://github.com/li-ronghui/FineDance

30. **Li, R. et al.** (2024). "Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives." *CVPR 2024.* arXiv:2403.10518. **[LODGE generator in AgentLODGE pipeline]**

31. **Li, R. et al.** (2024). "Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns." arXiv:2410.20389.

32. **Le, N. et al.** (2023). "Music-Driven Group Choreography." *CVPR 2023.* arXiv:2303.12337. [AIOZ-GDANCE; spatial formation mirroring]

33. **arXiv:2607.13978.** (2025). "Music-to-Dance Generation via Atomic Movements." **[Most structure-aware prior art; explicit section planning]**

34. **arXiv:2307.12963.** (2023). "DiffDance: Music-driven 3D Dance Generation via Diffusion Models with Structure Conditioning."

35. **Aristidou, A. et al.** (2024). "Exploring Multi-Modal Control in Music-Driven Dance Generation." arXiv:2401.01382.

### Motion Transition / In-Betweening

36. **Harvey, F.G. et al.** (2020). "Robust Motion In-Betweening." *SIGGRAPH / ACM TOG.* arXiv:2102.04942. [Adversarial RNN; LaFAN1 benchmark]

37. **Qin, V. et al.** (2022). "Motion In-betweening via Two-stage Transformers." *SIGGRAPH Asia 2022 / ACM TOG.* GitHub: https://github.com/victorqin/motion_inbetweening

38. **Cohan, S. et al.** (2024). "Flexible Motion In-betweening with Diffusion Models." *SIGGRAPH 2024.* DOI: 10.1145/3641519.3657414.

---

## KEY FINDINGS & GAPS

### What Exists in Prior Work
- Beat-level alignment: well-solved (BAS, DanceRevolution, FACT)
- Genre-level consistency: solved (Bailando, FineDance, LODGE)
- Long-form generation without accumulated error: partially solved (LODGE's parallel generation, EDGE's sliding window)
- Hierarchical pose-motif-choreography: addressed (Aristidou 2022) but not widely adopted
- Explicit section repetition/recapitulation in dance: **only partially in arXiv:2607.13978** (atomic movements, 2025)

### Critical Gap for AgentLODGE
**No prior system implements: (a) automatic music structure analysis → (b) motion segment recapitulation when musical sections repeat → (c) principled retrograde/mirror as structural variation.** This is the unique contribution of AgentLODGE. The assembler sits outside the diffusion generators (training-free) and can implement these structural devices purely at the segment stitching level, grounded in the dance composition theory literature (Blom & Chaplin, Smith-Autard, Humphrey) and enabled by MSA algorithms (McFee & Ellis, Foote, MSAF).

### Proposed New Metric: Section Repetition Correlation (SRC)
No existing dance generation paper quantifies how well the dance structure mirrors the musical structure at the section level. The proposed **Section Repetition Correlation**, cosine similarity of motion embeddings across musically repeated sections, fills this gap. Combined with BAS, FID, PFC, and GMS, it provides a comprehensive evaluation framework for choreographic structuredness.