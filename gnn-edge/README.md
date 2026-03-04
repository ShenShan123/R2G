# R2G: Multi-View Circuit Graph Benchmark Suite

## Overview

This repository contains the implementation for the R2G (RTL-to-GDSII) benchmark suite presented in CVPR 2026. It provides standardized multi-view circuit graph representations for placement and routing tasks in electronic design automation (EDA).

### Key Features

- **Multi-View Graph Representations**: Five complementary views of the same circuits
- **Standardized Data Pipeline**: From DEF files to homogeneous graphs with typed features
- **Reproducible Baselines**: Classic GNN models (GINE, GAT, ResGatedGCN) with fair comparison
- **Stage-Aware Tasks**: Node-level placement and edge-level routing predictions
- **Information Parity**: Same information content across all views for fair comparison

---

## Project Structure

```
.
├── main.py                  # Entry point: argument parsing and training pipeline
├── model.py                 # GNN model architecture (encoder + GNN + decoder head)
├── dataset.py               # Dataset class: loading, preprocessing, normalization
├── encoders.py              # Feature encoders for nodes and edges
├── sampling.py              # Data sampling and train/val/test split creation
├── downstream_train.py       # Training loop, evaluation, and checkpointing
└── data/                    # Data directory (not included)
    ├── place_merged_B_homograph/
    ├── place_merged_C_homograph/
    ├── ...
    └── route_merged_X_homograph/
```

---

## File-by-File Documentation

### 1. `main.py` - Entry Point

**Purpose**: Orchestrate the entire training pipeline from argument parsing to model training.

**Key Functions**:
- `parse_args()`: Parse all command-line arguments for model configuration
- `Tee`: Utility class to duplicate output to both console and log file
- `main()`: Main training pipeline

**Data Flow**:
```
Command-line arguments
       ↓
parse_args() → args object
       ↓
Setup logging (Tee to file)
       ↓
Set random seeds
       ↓
Load dataset (MergedHomographDataset)
       ↓
Sample data (dataset_sampling)
       ↓
Create model (GraphHead)
       ↓
Train model (downstream_train)
       ↓
Save results and logs
```

**Key Components**:

1. **Argument Groups**:
   - Data arguments: dataset selection, task level, task type
   - Sampling arguments: neighbor sampling parameters
   - Model arguments: GNN type, dimensions, layers
   - Training arguments: epochs, learning rate, optimizer
   - Encoder arguments: node/edge encoder types

2. **Logging Setup**:
   - Uses `Tee` class to write to both stdout and log file
   - Log file format: `{num_head_layers}_{dataset}_{num_gnn_layers}_{act_fn}_{src_dst_agg}_{model}_{lr}_{timestamp}.log`

3. **Random Seed Control**:
   - Ensures reproducibility across runs
   - Sets seeds for PyTorch, CUDA, and cuDNN

**Usage Example**:
```bash
python main.py \
    --dataset place_merged_B_homograph \
    --task_level edge \
    --task regression \
    --model gine \
    --num_gnn_layers 4 \
    --num_head_layers 2 \
    --hid_dim 256 \
    --lr 0.0001 \
    --epochs 100 \
    --gpu 0
```

---

### 2. `model.py` - GNN Architecture

**Purpose**: Implement the complete GNN model for circuit graph prediction tasks.

**Key Class**: `GraphHead`

**Architecture Components**:
```
Input Batch (raw features)
       ↓
FeatureEncoder (node + edge encoders)
       ↓
Encoded embeddings
       ↓
GNN Layers (message passing, K layers)
       ↓
Node embeddings
       ↓
Task-specific processing
    ├─ Node tasks: head_layers(node_emb)
    ├─ Edge tasks: aggregate(src, dst, edge) → head_layers()
    └─ Graph tasks: pool → head_layers()
       ↓
Predictions
```

**Forward Pass Details**:

**For Node-Level Tasks**:
1. Encode features → `[num_nodes, hidden_dim]`
2. Apply GNN layers → `[num_nodes, hidden_dim]`
3. Apply head MLP → `[num_nodes, output_dim]`
4. Filter valid nodes (labels != -1)

**For Edge-Level Tasks**:
1. Encode features → `[num_nodes, hidden_dim]`, `[num_edges, hidden_dim]`
2. Apply GNN layers → `[num_nodes, hidden_dim]`
3. Gather embeddings for labeled edges:
   - `src_emb = x[edge_label_index[0]]`
   - `dst_emb = x[edge_label_index[1]]`
   - `edge_attr_emb` (lookup from edge embeddings)
4. Aggregate (concat/add/mean) → `[num_labeled_edges, 2*hidden_dim]` or `3*hidden_dim`
5. Apply head MLP → predictions

**For Graph-Level Tasks**:
1. Encode features → `[num_nodes, hidden_dim]`
2. Apply GNN layers → `[num_nodes, hidden_dim]`
3. Global pooling (add/mean) → `[num_graphs, hidden_dim]`
4. Apply head MLP → predictions

**Supported GNN Types**:
- `gcn`: Graph Convolutional Network
- `sage`: GraphSAGE
- `gat`: Graph Attention Network (with edge features)
- `gine`: Graph Isomorphism Network with MLP (with edge features)
- `gated_gcn`: Residual Gated Graph Convolution (with edge features)

---

### 3. `dataset.py` - Data Loading and Preprocessing

**Purpose**: Load, preprocess, and normalize circuit graph data.

**Key Class**: `MergedHomographDataset`

**Data Processing Pipeline**:
```
Raw .pt file (x, y, edge_index, edge_attr, global_features, global_y)
       ↓
Load components
       ↓
Determine train/test masks (based on graph_ids)
       ↓
Apply normalization (regression only):
   1. Compute offset (if min <= 0)
   2. Log transform: log(x + offset + epsilon)
   3. Standardize: (x - mean) / std
   4. Save params for inverse transform
       ↓
Concatenate global features:
   - Node: [9 dims] + [6 dims] → 15 dims (14 in code, first 6 used)
   - Edge: [9 dims] + [6 dims] → 15 dims (14 in code, first 6 used)
       ↓
Create train/val/test masks
       ↓
Save processed data (cached for fast reload)
```

**Key Features**:

1. **Normalization Strategy**:
   - Only uses training data to compute mean/std (prevents data leakage)
   - Handles zero/negative values with offset
   - Supports both forward and inverse transforms

2. **Data Splits**:
   - Test set: Fixed 8 designs (IDs: 6, 7, 13, 15, 20, 25, 27, 29)
   - Validation set: Fixed 5 designs (IDs: 5, 14, 21, 24, 26)
   - Training set: Remaining designs

3. **Valid Label Handling**:
   - Uses sentinel value -1 for invalid/no-label samples
   - Only samples with labels != -1 are used for training/evaluation
   - `valid_mask` tracks which samples are valid

4. **Caching**:
   - Processed data is saved to disk
   - Subsequent runs load cached data (much faster)

**Node Feature Schema** (after processing):
- Columns 0-7: Original node features (coordinates, type, area, etc.)
- Column 8: Node type (0=gate, 1=io_pin, 2=net, 3=pin)
- Columns 9-13: Global features (design-level statistics)

**Edge Feature Schema** (after processing):
- Columns 0-7: Original edge features (connectivity info)
- Column 8: Edge type (0-8 for different edge types)
- Columns 9-13: Global features

---

### 4. `encoders.py` - Feature Encoding

**Purpose**: Transform raw circuit features into learned embeddings.

**Key Classes**:

1. **`HomographNodeEncoder`**:
   - Processes node features separately for each node type
   - Handles discrete (categorical) and continuous features differently
   - Discrete: Embedding lookup (learned vectors)
   - Continuous: Linear projection

2. **`HomographEdgeEncoder`**:
   - Similar to node encoder but for edge features
   - Processes each edge type separately

3. **`FeatureEncoder`**:
   - Combines node and edge encoders
   - Applies optional batch normalization

**Encoding Process for Nodes**:
```
Raw node features [num_nodes, feature_dim]
       ↓
Group by node_type
       ↓
For each node type:
   ├─ Discrete features: Embedding lookup → [num_nodes_type, emb_dim]
   └─ Continuous features: Linear projection → [num_nodes_type, emb_dim]
       ↓
Sum discrete + continuous
       ↓
Combine all node types
       ↓
Encoded embeddings [num_nodes, hidden_dim]
```

**Discrete vs Continuous Features**:

**Discrete (Categorical)**:
- Examples: cell_type (96 categories), orientation (8 categories), net_type (6 categories)
- Encoded via: `nn.Embedding(num_categories, embedding_dim)`
- Learned representations for each category

**Continuous (Numerical)**:
- Examples: x, y coordinates, area, power leakage
- Encoded via: `nn.Linear(input_dim, hidden_dim)`
- Linear transformation + learned weight/bias

**Node/Edge Type Processing**:
- Different node/edge types have different feature schemas
- Each type has its own embedding/projection layers
- This allows type-specific feature processing

**Task-Level Adaptation**:
- Node-level tasks: Include global features (columns 9-13)
- Edge-level tasks: Include global features (columns 9-13)
- The encoder adapts based on `task_level` parameter

---

### 5. `sampling.py` - Data Sampling

**Purpose**: Create train/val/test data loaders with neighbor sampling for GNN training.

**Key Functions**:

1. **`dataset_sampling()`**: Main function to create data loaders
2. **`get_nodes_from_subgraphs()`**: Sample nodes from specified graphs
3. **`sample_edges_by_graph_ids()`**: Sample edges from specified graphs

**Data Flow**:
```
MergedHomographDataset (full graph with all designs)
       ↓
Split graphs into train/val/test sets
       ↓
Sample nodes or edges:
   ├─ Node tasks: Sample nodes based on graph_ids and valid_mask
   └─ Edge tasks: Sample edges based on graph_ids and valid_mask
       ↓
Create data loaders:
   ├─ Node tasks: NeighborLoader (sample neighborhoods)
   └─ Edge tasks: LinkNeighborLoader (sample edge neighborhoods)
       ↓
Return loaders + max_label
```

**Neighbor Sampling**:

**Why Sample?**
- Large circuit graphs have millions of nodes/edges
- Full-batch training is infeasible
- Neighbor sampling creates manageable subgraphs for each batch

**How It Works (Node Tasks)**:
```
Target nodes (e.g., 512 nodes for batch)
       ↓
Sample neighbors within K hops:
   ├─ Hop 1: Sample N neighbors for each target node
   ├─ Hop 2: Sample N neighbors for each hop-1 node
   └─ ...
       ↓
Create subgraph containing target nodes + sampled neighbors
       ↓
Process only this subgraph in GNN
```

**Edge Task Sampling**:
- Base graph: Undirected (for message passing)
- Supervision edges: Directed (labeled edges)
- `LinkNeighborLoader` samples neighborhoods around supervision edges

**Sampling Parameters**:
- `num_hops` (default 2): Number of neighbor hops to sample
- `num_neighbors` (default 10): Number of neighbors to sample per hop
- `sample_ratio` (default 1.0): Fraction of nodes/edges to use
- `batch_size` (default 512): Number of targets per batch

**Graph Split**:
- Training graphs: Designs not in test set (19 designs)
- Validation graphs: Fixed 5 designs
- Test graphs: Fixed 8 designs
- Ensures no information leakage between splits

---

### 6. `downstream_train.py` - Training and Evaluation

**Purpose**: Implement training loop, evaluation metrics, and model checkpointing.

**Key Functions**:

1. **`train_epoch()`**: Train for one epoch
2. **`evaluate()`**: Evaluate model on a dataset
3. **`downstream_train()`**: Main training function with early stopping

**Training Loop**:
```
Initialize model, optimizer, scheduler, criterion
       ↓
For each epoch:
   ├─ Train for one epoch (train_epoch)
   │   └─ For each batch:
   │       ├─ Forward pass
   │       ├─ Compute loss (only on valid labels)
   │       ├─ Backward pass
   │       └─ Update weights
   ├─ Evaluate on validation set (evaluate)
   │   └─ Compute metrics (MAE, RMSE, R² for regression)
   ├─ Update learning rate (scheduler.step())
   └─ If validation improves:
       ├─ Evaluate on train and test sets
       ├─ Save model checkpoint
       └─ Record best metrics
       ↓
Return best test metrics
```

**Training Details**:

**Loss Functions**:
- Regression: `SmoothL1Loss` (Huber loss - robust to outliers)
- Classification: `CrossEntropyLoss`

**Optimization**:
- Optimizer: Adam
- Learning Rate: 0.0001 (default)
- Weight Decay: 1e-5 (L2 regularization)
- Scheduler: StepLR (halve every 20 epochs) or CosineAnnealingWarmRestarts

**Early Stopping**:
- Monitors validation loss
- Saves model when validation loss improves
- Returns best test metrics from the best validation checkpoint

**Evaluation Metrics**:

**Regression**:
- MAE (Mean Absolute Error): Average absolute difference
- RMSE (Root Mean Squared Error): Square root of average squared error
- R² (Coefficient of Determination): Proportion of variance explained

**Classification**:
- Accuracy: Proportion of correct predictions
- F1 Score: Weighted average of precision and recall
- Classification Report: Per-class precision, recall, F1

**Inverse Transform (Regression)**:
- Predictions are in normalized space
- For evaluation, convert back to original space:
  1. Inverse standardize: `x * std + mean`
  2. Inverse log: `exp(x) - epsilon`
  3. Remove offset: `x - offset`

---

## Data Flow Diagram

### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                           │
│  1. Parse arguments                                       │
│  2. Setup logging (Tee to file)                           │
│  3. Set random seeds                                      │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    dataset.py                              │
│  1. Load raw data from .pt file                          │
│  2. Split train/val/test (by graph_id)                   │
│  3. Normalize labels (log + standardize, train only)       │
│  4. Concatenate global features                             │
│  5. Create valid mask (labels != -1)                       │
│  6. Cache processed data                                    │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    sampling.py                             │
│  1. Sample nodes/edges from train/val/test graphs        │
│  2. Create NeighborLoader or LinkNeighborLoader           │
│  3. Print label distribution statistics                   │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                     model.py                               │
│  ┌─────────────────────────────────────────────┐            │
│  │              FeatureEncoder                │            │
│  │  ├─ HomographNodeEncoder (encoders.py)    │            │
│  │  └─ HomographEdgeEncoder (encoders.py)    │            │
│  └─────────────────┬───────────────────────────┘            │
│                    ↓                                   │
│  │            GNN Layers (K layers)        │            │
│  │  (message passing: GCN/SAGE/GAT/GINE/Gated) │         │
│  └─────────────────┬───────────────────────────┘            │
│                    ↓                                   │
│  │           Decoder Head (MLP)        │            │
│  │  (predicts labels for node/edge/graph)         │            │
│  └─────────────────┬───────────────────────────┘            │
│                    ↓                                   │
│              Predictions                              │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                downstream_train.py                          │
│  1. Setup optimizer, scheduler, criterion                  │
│  2. Training loop:                                        │
│     ├─ train_epoch: Forward → Loss → Backward → Update    │
│     ├─ evaluate: Compute metrics on val set              │
│     └─ Save best model (by val loss)                    │
│  3. Evaluate best model on test set                       │
│  4. Save results to .npz file                             │
└────────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
              Results and Logs
```

---

## Module Interactions

### Function Call Hierarchy

```
main()
├── parse_args()
├── MergedHomographDataset.__init__() [dataset.py]
│   └── MergedHomographDataset.process() [dataset.py]
│       ├── log_transform() [dataset.py]
│       ├── standardize() [dataset.py]
│       └── (save to cache)
├── dataset_sampling() [sampling.py]
│   ├── print_distribution_stats() [sampling.py]
│   ├── get_nodes_from_subgraphs() [sampling.py]
│   ├── sample_edges_by_graph_ids() [sampling.py]
│   ├── NeighborLoader() [PyG]
│   └── LinkNeighborLoader() [PyG]
├── GraphHead.__init__() [model.py]
│   └── FeatureEncoder.__init__() [encoders.py]
│       ├── HomographNodeEncoder.__init__() [encoders.py]
│       └── HomographEdgeEncoder.__init__() [encoders.py]
├── downstream_train() [downstream_train.py]
│   ├── train_epoch() [downstream_train.py]
│   │   └── GraphHead.forward() [model.py]
│   │       └── FeatureEncoder.forward() [encoders.py]
│   │           ├── HomographNodeEncoder.forward() [encoders.py]
│   │           └── HomographEdgeEncoder.forward() [encoders.py]
│   └── evaluate() [downstream_train.py]
│       ├── GraphHead.forward() [model.py]
│       └── inverse_standardize(), inverse_log_transform() [dataset.py]
└── (save results)
```

### Data Transformations

1. **Raw Data → Processed Data** [dataset.py]:
   ```
   x: [N, 10] → [N, 14] (concatenate global features)
   y: [N,] → [N,] (normalized for regression)
   edge_index: [2, E] → [2, E] (unchanged)
   edge_attr: [E, 10] → [E, 14] (concatenate global features)
   ```

2. **Raw Features → Embeddings** [encoders.py]:
   ```
   Node features: [N, feature_dim] → [N, hidden_dim]
   Edge features: [E, feature_dim] → [E, hidden_dim]
   ```

3. **Embeddings → Predictions** [model.py]:
   ```
   Node tasks: [N, hidden_dim] → [N, 1] (regression) or [N, num_classes] (classification)
   Edge tasks: [E, 2*hidden] → [E, 1] (regression)
   ```

4. **Normalized → Original Scale** [dataset.py, downstream_train.py]:
   ```
   y_norm → y_original:
     1. y_std = y_norm * std + mean
     2. y_log = exp(y_std) - epsilon
     3. y_orig = y_log - offset
   ```

---

## How to Reproduce Experiments

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install torch torch-geometric numpy scikit-learn matplotlib tqdm
   ```

2. **Prepare Data**:
   - Place processed .pt files in `data/{dataset_name}/raw/`
   - Dataset naming: `place_merged_{VIEW}_homograph.pt` or `route_merged_{VIEW}_homograph.pt`
   - Views: B, C, D, E, F

3. **Run a Single Experiment**:
   ```bash
   python main.py \
       --dataset place_merged_B_homograph \
       --task_level edge \
       --model gine \
       --num_gnn_layers 4 \
       --num_head_layers 2 \
       --epochs 100 \
       --gpu 0
   ```

To run experiments for different views or model configurations, modify the `--dataset`, `--model`, `--num_gnn_layers`, and `--num_head_layers` parameters accordingly.

### Understanding Output

**Log Files** (`logs/`):
- Contains complete training progress
- Format: `{num_head_layers}_{dataset}_{num_gnn_layers}_{act_fn}_{src_dst_agg}_{model}_{lr}_{timestamp}.log`
- Includes: configuration, epoch-by-epoch metrics, final test results

**Model Checkpoints** (`results/{dataset}_{model}_{task_level}/`):
- `best_model.pt`: Best model weights (by validation loss)

**Test Results** (`results/{dataset}_{model}_{task_level}/`):
- `test_results.npz`: NumPy array with predictions and labels

**Key Metrics to Check**:
- Train/Val/Test Loss: Should decrease over epochs
- Test R²: Higher is better (closer to 1.0)
- Test MAE/RMSE: Lower is better

---

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory
**Cause**: Large batch size or graph size
**Solution**: Reduce `--batch_size` or `--sample_ratio`

### Issue 2: Training Diverges (NaN loss)
**Cause**: Learning rate too high or invalid data
**Solution**:
- Lower `--lr` (try 1e-5)
- Check data for invalid values
- Reduce `--num_gnn_layers` or `--hid_dim`

### Issue 3: Poor Test Performance
**Cause**: Overfitting or inappropriate model/view
**Solution**:
- Increase dropout (`--dropout 0.5`)
- Add weight decay (`--weight_decay 1e-4`)
- Try different view or model

### Issue 4: Slow Training
**Cause**: Large graphs, many epochs, or CPU training
**Solution**:
- Use GPU (`--gpu 0`)
- Reduce epochs for testing (`--epochs 10`)
- Increase `--num_workers` for data loading
- Reduce `--num_neighbors` or `--num_hops`

### Issue 5: Results Don't Match Paper
**Cause**: Different random seed, data split, or hyperparameters
**Solution**:
- Ensure `--seed 42` (default)
- Verify test_graph_ids in `dataset.py` match paper
- Check all hyperparameters match paper configuration
- Confirm data preprocessing is identical

---

## Contact

For questions about the code, datasets, or experiments, please contact the authors.
