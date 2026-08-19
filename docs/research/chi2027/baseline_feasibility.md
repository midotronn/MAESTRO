# Existing-method comparison feasibility

This screening freezes the first technical pilot around **Bailando++** and **FineDance**, alongside
the already integrated LODGE and EDGE baselines. Beat-It remains related work only because its
official repository does not provide runnable inference code or pretrained weights.

## Decision

| Method | Technical pilot | Publication gate | Main conversion work |
|---|---|---|---|
| Bailando++ | Yes | Non-commercial NTU S-Lab license is compatible with research evaluation | 60 to 30 FPS, rotation-matrix conversion, SMPL-24 body mapping |
| FineDance | Yes | Confirm the unstated repository license before publishing generated artifacts | SMPL-H body subset, 6D rotation conversion, root/contact validation |
| Beat-It | No | No code, weights, or license are available | Not applicable |

## Bailando++

- Official repository: <https://github.com/lisiyao21/Bailando>
- Bailando paper: <https://arxiv.org/abs/2203.13055>
- Bailando++ paper: <https://ieeexplore.ieee.org/document/10264209>
- Inference code and pretrained weights are linked by the official repository.
- Native motion is based on the AIST++ 60 FPS convention. MAESTRO must freeze and test one 60 to
  30 FPS conversion policy before comparison.
- The repository depends on an older Python/PyTorch stack, `essentia`, `chumpy`, and a separately
  licensed SMPL model.
- The official license is the NTU S-Lab License 1.0 for non-commercial research use.

## FineDance

- Official repository: <https://github.com/li-ronghui/FineDance>
- Paper: <https://arxiv.org/abs/2212.03741>
- Inference code and pretrained weights are linked by the official repository.
- Custom WAV inference uses 35-dimensional librosa features and outputs motion natively at 30 FPS.
- The output is a 319-dimensional SMPL-H representation. The comparison adapter must remove
  hand-only joints for Y-Bot, convert 6D rotations, and validate root scale and contacts.
- The published environment pins a CPU-only PyTorch package despite requiring GPU inference. The
  pilot environment must override that pin explicitly.
- The repository does not state a license. Internal reproducibility work can proceed, but public
  redistribution or publication of generated assets must wait for license confirmation.

## Beat-It

- Official repository: <https://github.com/ZikaiHuangSCUT/Beat-It>
- Paper: <https://arxiv.org/abs/2407.07554>
- Project page: <https://zikaihuangscut.github.io/Beat-It/>
- As screened, the official repository contains a placeholder rather than inference code, weights,
  or a license. Reimplementing the method from the paper would not be a fair reproducibility
  comparison.

## Pilot gates

Neither selected method enters the frozen comparison corpus until it passes:

1. clean environment construction;
2. official-weight checksum capture;
3. one fixed-song inference;
4. representation conversion tests;
5. deterministic common Y-Bot rendering;
6. duration, root, joint, ground, and invalid-value checks; and
7. a run manifest containing the exact code revision and configuration.

The protocol is stored under `experiments/comparisons/`.
