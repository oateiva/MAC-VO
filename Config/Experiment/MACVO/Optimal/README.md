# Optimal configurations — depth model × backend

Best-known settings for each {DAv2, DAv3} × {GTSAM, G-EDF} combination, distilled from the
sweeps in [`ProgressReports/`](../../../../ProgressReports/README.md) (updated 2026-07-08).
All numbers are ATE RMSE (m), Sim(3) scale-corrected + aligned
(`EvaluateSequences(align=True, correct_scale=True)`), on EIVA `plane_nose[80:160]`.

| Config | Backend | Key deviations from defaults | ATE RMSE (m) | Seeds |
|---|---|---|--:|---|
| [`MACVO_DAv2_GTSAM.yaml`](MACVO_DAv2_GTSAM.yaml) 🏆 | `GTSAM_Graph` pose2point, SE(3) | tuned combo: `huber 0.05, hprev 3.0, atol 2.0, psigma 1e-3` | **0.233 / 0.229** | 2 |
| [`MACVO_DAv2_GEDF.yaml`](MACVO_DAv2_GEDF.yaml) | `GEDF_PGO` `gedf+icp` | `alignment: sim3 (w=100)`, `autodiff: true`, `field.sigma 0.20` | 0.251 | 1 |
| [`MACVO_DAv3_GEDF.yaml`](MACVO_DAv3_GEDF.yaml) | `GEDF_PGO` `gedf+icp`, SE(3) | defaults (`sigma 0.30`); `device_depth: cuda:1` | 0.618 | 1 |
| [`MACVO_DAv3_GTSAM.yaml`](MACVO_DAv3_GTSAM.yaml) | `GTSAM_Graph` pose2point, SE(3) | optimizer defaults (tuned DAv2 combo transfers *negatively*) | 0.689 | 1 |
| [`MACVO_Fast_GEDF.yaml`](MACVO_Fast_GEDF.yaml) | `GTSAM_Graph` `pose2point+gedf`, SE(3) | **stereo** Fast frontend; GTSAM ICP + G-EDF field in one joint solve; fine 0.2 m grid; stereo insert gate `0.0675` | *untested* | — |
| [`MACVO_Fast_GEDF_Pure.yaml`](MACVO_Fast_GEDF_Pure.yaml) | `GEDF_PGO` `gedf` (field only), SE(3) | **stereo** Fast frontend; NO ICP factor; mapper identical to the hybrid (isolates the ICP factor) | *untested* | — |

All four set optimizer `parallel: false` — the sweeps showed the 1 s parallel timeout drops
results nondeterministically; sequential solving reproduces the reported numbers. Flip it back
to `true` for pipelined speed if determinism doesn't matter.

## Rules these configs encode

1. **Nothing transfers between depth models** — the tuned DAv2 GTSAM combo *hurts* DAv3;
   G-EDF sigma optima differ per frontend. Re-tune per depth model.
2. **Alignment (Sim(3)) helps exactly one cell**: DAv2 + G-EDF (−34 %). On GTSAM the landmarks
   already absorb scale error (sim3 neutral-to-worse, sl4 seed-unstable); on DAv3 the depth is
   already scale-consistent (any warp hurts). SL(4) overfits everywhere.
3. **The ICP factor is load-bearing in G-EDF** — pure field registration (`graph_type: gedf`)
   stays ≥ 0.67 even after tuning every mapper knob.
4. **DAv3 caveats**: needs `device_depth: cuda:1` under G-EDF (single 24 GB card OOMs), and is
   optimizer-bound under GTSAM (~20 s/frame at `num_point 1000`).

## Caveats

- Every number comes from one 80-frame `plane_nose` segment; the two G-EDF/DAv3 entries are
  single-seed. Directional, not a paper-grade multi-sequence average.
- Reports: [optimizer sweep](../../../../ProgressReports/2026-07-06_optimizer-hyperparameter-sweep.md) ·
  [GEDF alignment benchmark](../../../../ProgressReports/2026-07-07_gedf-alignment-manifold-benchmark.md) ·
  [GTSAM alignment sweep](../../../../ProgressReports/2026-07-07_gtsam-alignment-sweep.md) ·
  [pure-GEDF mapper sweep](../../../../ProgressReports/2026-07-07_pure-gedf-mapper-sweep.md)

## Run

```bash
python MACVO.py --odom Config/Experiment/MACVO/Optimal/MACVO_DAv2_GTSAM.yaml \
                --data Config/Sequence/EIVA_Dataset/plane_nose.yaml --useRR
```
