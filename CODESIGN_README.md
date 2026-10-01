# LeWM FPGA Co-Design Experiment Extensions

This fork works with the companion modified `stable-worldmodel` fork.

## Implemented experiment features

- FP16 / W8A16 / W8A8 algorithmic precision modes;
- `quant_scope = predictor | encoder | all`;
- complete CEM trace export;
- Sample tiers: `64 / 96 / 128 / 176 / 208 / 256 / 300`;
- CEM prefix validation;
- fixed-S OFAT replay;
- same-candidate precision rescoring;
- suffix replay for per-iteration Sample-tier calibration;
- Direct Tier Policy support (`K_t,U_t -> S_(t+1)`);
- optional hysteresis, disabled by default;
- ranking metrics and conservative tier-threshold fitting.

## Quantization meaning

W8A16/W8A8 are numerical-fidelity experiments. Weights are symmetric 8-bit
quantize/dequantize; W8A8 also fake-quantizes activations. CUDA execution uses FP16
autocast. These modes must not be reported as native GPU INT8 speedups.

## Default behavior

The evaluation YAMLs contain a `codesign:` block. Initially use:

```yaml
codesign:
  precision: fp16
  quant_scope: predictor
  trace_enabled: true
  trace_candidates: true
  adaptive_sample: false
  sample_tiers: [64, 96, 128, 176, 208, 256, 300]
  tier_thresholds: {}
  hysteresis: false
```

Do not enable `adaptive_sample` until thresholds have been calibrated from replay.

## Baseline / trace example

Use the normal LeWM evaluation entry point, for example:

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm \
  codesign.precision=fp16 codesign.adaptive_sample=false
```

The run saves `codesign_trace/trace.npz`, `metadata.json`, and `config.yaml` beside the
normal evaluation output.

## Offline replay examples

All commands reuse the saved baseline planning state and do not step the simulator.

```bash
# Fixed-S OFAT: 64/96/128/176/208/256/300
python -m codesign.replay_ofat --config-name=pusht \
  policy=pusht/lewm \
  codesign.replay_trace=/path/to/codesign_trace/trace.npz \
  codesign.replay_mode=static_sample codesign.precision=fp16

# Per-iteration suffix replay
python -m codesign.replay_ofat --config-name=pusht \
  policy=pusht/lewm \
  codesign.replay_trace=/path/to/codesign_trace/trace.npz \
  codesign.replay_mode=suffix codesign.precision=fp16

# Same-candidate W8A16 rescore
python -m codesign.replay_ofat --config-name=pusht \
  policy=pusht/lewm \
  codesign.replay_trace=/path/to/codesign_trace/trace.npz \
  codesign.replay_mode=rescore codesign.precision=w8a16

# Prefix equivalence validation
python -m codesign.replay_ofat --config-name=pusht \
  policy=pusht/lewm \
  codesign.replay_trace=/path/to/codesign_trace/trace.npz \
  codesign.replay_mode=prefix_validate codesign.precision=fp16
```

## Tier labels and thresholds

```bash
python -m codesign.calibrate_tiers \
  --baseline /path/to/trace.npz \
  --replay /path/to/suffix_replay.npz \
  --action-threshold <ACTION_FIDELITY_THRESHOLD> \
  --energy-regret-threshold <OPTIONAL_ENERGY_THRESHOLD> \
  --output tier_labels.csv

python -m codesign.fit_tier_thresholds \
  --labels tier_labels.csv \
  --underalloc-quantile 0 \
  --output tier_thresholds.json
```

After validation, copy the fitted thresholds into `codesign.tier_thresholds` and set
`codesign.adaptive_sample=true`.
