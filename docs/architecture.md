# Architecture

## Overview

The pipeline processes a single car photo through seven stages, moving from
pixel-level computer vision to structured reasoning to natural-language
reporting.

## Stage details

### 1. Part Segmentation
YOLOv8-seg fine-tuned on [Carparts-seg](https://docs.ultralytics.com/datasets/segment/carparts-seg).
Outputs a binary mask per detected part.

### 2. Damage Segmentation
A separate YOLOv8-seg model fine-tuned on [CarDD](https://cardd-ustc.github.io).
Trained independently because it uses a different label space (damage type,
not part type). Outputs a binary mask + damage type per detection.

### 3. Fusion
Pure Python logic (`src/fusion/fusion.py`). Computes, for each part mask,
what fraction of each damage mask overlaps it (IoU-style). Damage is
attributed to a part when the overlap exceeds a configurable threshold
(default 0.5). No model is involved in this stage.

Class names from the two source datasets are reconciled via
`src/fusion/class_mapping.py`, since they were labeled independently and
don't share a naming convention.

### 4. Hidden Damage Risk Estimation
**Not a trained model.** Uses a hand-authored structural adjacency graph
(`src/hidden_damage_risk/structural_graph.py`) describing which parts are
physically connected, combined with the severity/area of visible damage,
to produce a heuristic risk percentage for damage to non-visible
components (wiring, frame, sensors).

This stage is fundamentally limited: no vision model can see through sheet
metal. The output should always be treated as a triage signal, not a
diagnosis. Every result includes `confidence` and `basis` fields.

### 5. Cost Estimation
A two-layer approach:
- **Offline**: a configurable regional price table (`configs/pricing_config.yaml`)
  with manually researched sample values.
- **Online (optional)**: Gemini API with Google Search grounding, which lets
  the model search current sources and synthesize a cost range with
  citations. Requires `GEMINI_API_KEY` and network access.

No free, reliable live pricing API exists for vehicle repair costs in any
market, so both layers should be presented as estimates, not quotes.

### 6. Repair vs. Replace Advisor
Weighs upfront repair cost against expected resale depreciation. High-
visibility structural panels (doors, hood, fenders) are penalized more
heavily when repaired, since repainted/reworked panels are detectable by
appraisers (e.g. via paint thickness gauges) and tend to reduce resale
value more than a full replacement would. Like stage 4, depreciation
figures are heuristic estimates.

### 7. Report Generation
Gemini API call that takes the full structured output from stages 1-6 and
produces a natural-language report, clearly separating measured facts from
estimated figures, and always closing with a reminder that the report is a
preliminary automated assessment, not a substitute for an in-person
inspection.

## Known limitations

- Hidden damage risk and resale depreciation are heuristic, not
  data-driven — there is no large-scale dataset of confirmed hidden damage
  or repair-vs-replace resale outcomes to train on.
- The two segmentation models are trained on different, independently
  labeled datasets, so class alignment relies on manual mapping
  (`src/fusion/class_mapping.py`) rather than a shared taxonomy.
- Cost figures depend on a manually maintained price table unless the
  optional Gemini + Google Search enrichment is used, which itself is a
  search-synthesized estimate, not a certified price.

## Possible extensions (not yet implemented)

- Fraud/tamper detection (e.g. Error Level Analysis) on the input photo.
- Multi-image support to aggregate damage across several angles of the
  same vehicle.
- Repair vs. total-loss classifier for severe multi-part damage.
- Human-in-the-loop flagging when model confidence is low.
