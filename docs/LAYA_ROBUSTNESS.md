# Laya robustness and the 85% target

## What the target means

TR8D does not claim that a model can predict every daily BUY/SELL/HOLD label at
85% accuracy. The deployable target is **at least 85% accuracy on the subset of
decisions Laya accepts**, with at least 10% coverage. Remaining decisions are
HOLD and are sent to the background LLM reviewer.

The confidence threshold is selected on the chronological calibration split.
It is then frozen and measured once on the later untouched test split. A model
is promoted only when all checks pass:

- at least 500 untouched test examples;
- selective accuracy at least 85%;
- selective coverage at least 10%;
- overall accuracy above the majority-class baseline;
- Brier score below the base checkpoint.

Accuracy, balanced accuracy, macro F1, per-class precision/recall, confusion
matrix, Brier score, log loss, calibration error, and coverage are all stored.
This prevents a model that always predicts HOLD from looking successful.

## Current audit

The first local checkpoint does **not** pass this gate. Its 78-example synthetic
test split is too small, and it primarily predicts HOLD:

- overall accuracy: 44.87%;
- majority baseline: 43.59%;
- balanced accuracy: 34.78%;
- macro F1: 23.20%;
- 85%-accuracy calibration policy: no qualifying coverage;
- production confidence threshold: 1.0, which safely routes decisions to the
  background LLM until a better checkpoint passes.

See `reports/laya-quality-report.json` for the complete audit.

A second run replaced the heuristic input with expanding-window logistic
predictions and used validation-only early stopping. It also failed: 26.98%
overall accuracy, 32.05% balanced accuracy, and no qualifying 85%-accuracy
coverage. That negative result is preserved in
`reports/laya-oof-training-report.json`; it confirms that the small synthetic
panel does not contain enough transferable signal. More epochs on it are not a
credible remedy.

## Data required for a serious run

The minimum 500-example test gate implies at least roughly 3,400 chronological
examples with the current 70/15/15 split. That is only a mechanical minimum.
A serious experiment should use substantially more observations across:

- multiple liquid stocks, sector ETFs, and a broad-market baseline;
- bull, bear, high-volatility, low-volatility, and rate-shock regimes;
- corporate-action-consistent open and close prices;
- delisted names where available, to reduce survivorship bias;
- evidence carrying exact `published_at`, `ingested_at`, and `available_at`;
- realistic spreads, fees, and slippage in downstream evaluation.

The final test date range must remain sealed. Do not repeatedly tune against it.
Create a newer forward test window after any material feature or label change.

## Recommended training sequence

1. Train and calibrate the numerical price model with expanding walk-forward
   folds. The default exporter now fits its logistic baseline only on earlier
   dates and gives Laya those out-of-sample probabilities. Laya is the decision
   policy, not a replacement for a numerical return model.
2. Export the point-in-time Laya dataset. TR8D writes both its native JSONL and
   `*_rlcd.jsonl` files compatible with the data shape used by Laya's official
   RLCD notebook. Each market outcome produces a flat-wallet and held-position
   example, so SELL is never taught when the wallet has no position.
3. Establish a head-only GPU baseline with the included Colab notebook.
4. Run full encoder + head RLCD on a dual-T4 Kaggle runtime or an equivalent
   multi-GPU Colab runtime. Use the official Laya notebook as the maintained
   reference implementation.
5. Fit temperature and the abstention threshold only on calibration data.
6. Run `evaluate-laya` once on the sealed test split. Deploy only if every gate
   passes; otherwise collect more data or improve the upstream price model.

## Commands

```bash
python -m tr8d export-laya-dataset prices.csv --output var/laya-training

python -m tr8d train-laya \
  --dataset var/laya-training \
  --output var/models/laya-tr8d \
  --device cuda --epochs 4 \
  --target-accuracy 0.85 --minimum-coverage 0.10 \
  --minimum-test-examples 500

python -m tr8d evaluate-laya \
  --dataset var/laya-training \
  --model var/models/laya-tr8d \
  --device cuda --write-policy \
  --output var/laya-quality-report.json
```

The runtime automatically reads `tr8d_policy.json` from a local checkpoint.
If the checkpoint cannot demonstrate the required selective accuracy, its
threshold is set to 1.0 and uncertain cases remain HOLD + background LLM.

Official full-RLCD reference:
https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
