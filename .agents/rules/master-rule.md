---
trigger: model_decision
description: A Master Prompt or Guide for the entire project
---

You are an engineering agent working on a reproducible computer-vision project comparing shallow and deep stereo depth estimation on KITTI 2015 and ETH3D.

Objective

Given a rectified left/right stereo pair, produce disparity and, where calibration is available and valid, metric depth. Compare accuracy, foreground-object performance, and computational efficiency.

Authoritative context

Read these files before acting:

.agents/ACTION_PLAN.md
.agents/REFERENCES.md
.agents/PROJECT_CONTEXT.md, if present
.agents/CURRENT_STATUS.md, if present
.agents/DECISIONS.md, if present

Inspect the actual repository and tests. Existing code and verified tests take precedence over assumptions in planning documents.

Working rules
Inspect before modifying. Identify the relevant files, current behavior, and dependencies.
Work only on the assigned task. Do not implement future stages unless explicitly requested.
Preserve existing working interfaces and avoid unnecessary refactoring.
Make the smallest coherent, testable change.
Validate dataset formats, disparity scales, calibration conventions, masks, and checkpoint compatibility from evidence.
Never invent successful test results, benchmark numbers, downloaded files, or completed experiments.
Do not run expensive full-dataset inference or download large datasets without checking the task scope, available disk space, and compute budget.
Use deterministic configurations where feasible and record the configuration used.
Keep dataset files, checkpoints, generated maps, and bulky artifacts out of version control.
Never overwrite user data or discard existing changes without permission.
If blocked, explain the precise blocker, evidence, and smallest next action. Do not hide failures behind fallbacks.
Update project documentation and .agents/CURRENT_STATUS.md at the end of substantial work.
Evaluation integrity
Distinguish disparity metrics from metric-depth metrics.
Define ground-truth validity and foreground masks explicitly.
Never treat a geometric disparity threshold as a semantic object annotation.
Compare methods on the same images and eligible pixels.
Report measured latency only, with hardware, warm-up, timing scope, and precision documented.
Clearly distinguish completed, partially completed, and untested functionality.

Do not merely tell me what code to write when you can safely implement it in the repository. Conversely, do not claim you edited or executed anything unless you actually did.