# Car Damage Assessment Pipeline

An end-to-end computer vision + LLM pipeline that assesses vehicle damage from a
photo — inspired by the workflows used by InsurTech companies (e.g. Tractable, CCC)
to speed up insurance damage appraisal.

Given a photo of a damaged car, the system identifies which parts are damaged,
estimates how severe the damage is, flags the risk of hidden/structural damage,
estimates repair cost, recommends repair vs. replacement, and produces a
human-readable report.

## Why this matters

Manual vehicle damage appraisal is slow, inconsistent, and requires a trained
inspector on-site. Automating the first pass — even imperfectly — lets insurers
and repair shops triage claims faster and lets private sellers/buyers get a
quick, explainable second opinion before an in-person inspection.

## Pipeline architecture

```
                              Input photo
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                     ▼
      Part Segmentation                    Damage Segmentation
      (which pixels = which part)          (which pixels = which damage)
                └─────────────────┬─────────────────┘
                                  ▼
                              Fusion
                  (overlaps masks via IoU → damage % per part)
                                  ▼
                    Hidden Damage Risk Estimator
        (structural adjacency graph + rule-based/LLM reasoning)
                                  ▼
                          Cost Estimation
                (regression + base price table, optionally
                 enriched with Gemini + Google Search grounding)
                                  ▼
                  Repair vs. Replace Advisor
        (weighs upfront cost against resale depreciation risk)
                                  ▼
                    Report Generation (Gemini API)
                  (natural-language summary for the end user)
```

## Stages

| # | Stage | What it does | Type |
|---|-------|---------------|------|
| 1 | Part Segmentation | Pixel-level mask of each car part (door, bumper, fender, hood...) | Deep learning model (YOLOv8-seg), trained on [Carparts-seg](https://docs.ultralytics.com/datasets/segment/carparts-seg) |
| 2 | Damage Segmentation | Pixel-level mask of each damage instance (dent, scratch, crack, glass shatter...) | Deep learning model (YOLOv8-seg), trained on [CarDD](https://cardd-ustc.github.io) |
| 3 | Fusion | Computes what % of each part's area is damaged, by overlapping the two mask sets (IoU) | Deterministic logic, no learning |
| 4 | Hidden Damage Risk | Estimates the probability that visible damage propagated to non-visible parts (wiring, frame, sensors) | Rule-based structural graph + LLM reasoning (not a trained model — see caveat below) |
| 5 | Cost Estimation | Estimates repair cost from damage type/severity/area | Regression model + configurable price table, optionally enriched with Gemini + Google Search grounding |
| 6 | Repair vs. Replace Advisor | Recommends repair vs. full part replacement, weighing upfront cost against expected resale value depreciation | Rule-based reasoning + optional LLM/search enrichment |
| 7 | Report Generation | Turns all structured outputs into a readable report | Gemini API |

> ⚠️ **Important limitation**: stages 4 and 6 produce **probabilistic estimates**,
> not certified measurements. No vision model can see inside a door panel or
> under a bumper from an external photo — hidden damage risk and resale
> depreciation are engineering/market heuristics, not ground truth. These
> numbers are meant to triage and inform, not replace an in-person inspection
> by a certified appraiser. Every hidden-risk and depreciation output includes
> a `confidence` and `basis` field so the reasoning stays transparent to the
> end user.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Project structure

```
car-damage-assessment/
├── data/
│   ├── raw/                    # raw datasets (gitignored, download separately)
│   └── processed/              # processed data ready for training
├── src/
│   ├── part_segmentation/      # part segmentation model train/inference
│   ├── damage_segmentation/    # damage segmentation model train/inference
│   ├── fusion/                 # mask fusion logic
│   ├── hidden_damage_risk/     # structural graph + risk estimator
│   ├── cost_estimation/        # repair cost regression + price table
│   ├── repair_advisor/         # repair vs. replace recommendation
│   ├── report_generation/      # Gemini API report generation
│   └── pipeline.py             # runs the full pipeline end to end
├── api/
│   └── main.py                 # FastAPI endpoint
├── configs/
│   ├── config.yaml             # model paths, thresholds, etc.
│   └── pricing_config.yaml     # regional base price table
├── tests/
├── notebooks/                   # EDA / experimentation notebooks
└── docs/
    └── architecture.md          # full technical write-up
```

## Usage

```bash
python -m src.pipeline --image path/to/car_photo.jpg
```

## Datasets

- **Damage segmentation**: [CarDD](https://cardd-ustc.github.io) — 4,000 high-res
  images, 9,000+ annotated instances across 6 damage categories. Requires filling
  a short license form to get the download link.
- **Part segmentation**: [Ultralytics Carparts-seg](https://docs.ultralytics.com/datasets/segment/carparts-seg) —
  ready-to-use YOLO segmentation dataset, 3,516 train images, 19 part classes.

Because these datasets were built independently, part class names don't match
1:1 (e.g. `front_door` vs `Front-door`). See `src/fusion/class_mapping.py` for
the normalization table used to reconcile them.

## Roadmap

- [x] Project scaffolding
- [ ] Class name mapping between the two datasets
- [ ] Train part segmentation model
- [ ] Train damage segmentation model
- [ ] Fusion logic
- [ ] Structural adjacency graph + hidden damage risk estimator
- [ ] Cost estimation model
- [ ] Repair vs. replace advisor
- [ ] Gemini report generation
- [ ] FastAPI endpoint
- [ ] Tests
- [ ] Docker deployment
- [ ] Public demo (Hugging Face Spaces / Streamlit)

## License

MIT
