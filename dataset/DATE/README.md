# DATE Graph Dataset

This directory is the dataset entry point for the ready-to-train DATE graph
artifacts used by R2G. The dataset covers placement and routing tasks in both
heterogeneous and homogeneous graph formats.

The dataset is organized by task and graph format:

```text
dataset/DATE/
├── placement/
│   ├── heterogeneous_graphs/
│   │   ├── placement_view_b_heterogeneous_graph.pt
│   │   ├── placement_view_c_heterogeneous_graph.pt
│   │   ├── placement_view_d_heterogeneous_graph.pt
│   │   ├── placement_view_e_heterogeneous_graph.pt
│   │   └── placement_view_f_heterogeneous_graph.pt
│   └── homogeneous_graphs/
│       ├── placement_view_b_homogeneous_graph.pt
│       ├── placement_view_c_homogeneous_graph.pt
│       ├── placement_view_d_homogeneous_graph.pt
│       ├── placement_view_e_homogeneous_graph.pt
│       └── placement_view_f_homogeneous_graph.pt
└── routing/
    ├── heterogeneous_graphs/
    │   ├── routing_view_b_heterogeneous_graph.pt
    │   ├── routing_view_c_heterogeneous_graph.pt
    │   ├── routing_view_d_heterogeneous_graph.pt
    │   ├── routing_view_e_heterogeneous_graph.pt
    │   └── routing_view_f_heterogeneous_graph.pt
    └── homogeneous_graphs/
        ├── routing_view_b_homogeneous_graph.pt
        ├── routing_view_c_homogeneous_graph.pt
        ├── routing_view_d_homogeneous_graph.pt
        ├── routing_view_e_homogeneous_graph.pt
        └── routing_view_f_homogeneous_graph.pt
```

See `MANIFEST.tsv` for the repository filename mapping, release package names,
and byte sizes.

## Release Packages

The binary graph artifacts are distributed as release assets:

- `r2g-date-placement-heterogeneous-graphs.tar.gz`
- `r2g-date-placement-homogeneous-graphs.tar.gz`
- `r2g-date-routing-heterogeneous-graphs.tar.gz`
- `r2g-date-routing-homogeneous-graphs.tar.gz`

The complete DATE graph dataset is approximately 3.7 GB, and many `.pt` files
are larger than GitHub's regular 100 MB file limit. Store the binary artifacts
with GitHub Releases or Git LFS, and keep this directory as the stable
in-repository dataset index.
