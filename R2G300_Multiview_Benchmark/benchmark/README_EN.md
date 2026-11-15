# R2G-300: Multi-view GNN baselines

This code-only package provides GAT, GINE, ResGatedGCN, HAN, HGT and RGCN for per-Net routed wirelength and per-Gate congestion prediction, with placement/route inputs and B–F graph views. The dataset has 30 families with 10 variants each. The family-disjoint fixed split has 240/30/30 instances.

See [README.md](README.md) for commands and tables. No dataset or weights are included. Pass an external multiview dataset root containing manifest.csv and samples/ through --data-root. The original project path is provenance, not a dataset root.

The v3 100-epoch seed42 matrix has 84 recorded results, 24 unmeasured placement-congestion entries and 12 unsupported E-view congestion entries. Historical L3/H2 two-stage results and v3 50-epoch three-seed Route results are separate cohorts, never merged to fill missing cells. The original GAT/C Route congestion test R² is 0.6911±0.0077 across seeds42/43/44 in the 50-epoch cohort.

We welcome improvements to typed feature encoding, edge-aware message passing, task-specific readout and explicitly aligned multi-view representations. Tune with validation data; report fixed splits, multiple training seeds, R²/MAE/RMSE and model size. Existing historical metrics are provided as evidence; this packaging does not claim fresh full retraining or bitwise cross-environment reproduction.

Only the six portable model implementations are retained in two folders. Historical placement metrics remain references; their older frozen source copies are excluded. All 300 v3 placement metadata records specify a floorplan feature cutoff without trusted standard-cell coordinates. This prevents spatial grid membership at input time, but does not remove the route-derived Gate labels. Earlier legacy placement congestion experiments did run (24 completed records); the v3 L4/H3 placement congestion results were not found.
