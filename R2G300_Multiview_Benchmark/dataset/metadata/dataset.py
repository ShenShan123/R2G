"""Lazy, manifest-based loading of aligned B/C/D/E/F graph samples."""
import csv
from pathlib import Path


class MultiViewDataset:
    """One item is one design/version/stage with selected native PyG views.

    Implements the map-style Dataset protocol accepted by torch DataLoader.
    Use collate_multiview when batching PyG graphs.
    """
    def __init__(self, root=None, views=('B', 'C', 'D', 'E', 'F'), stages=None, designs=None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parent
        self.views = (views,) if isinstance(views, str) else tuple(views)
        if not self.views or len(set(self.views)) != len(self.views) or set(self.views) - set('BCDEF'):
            raise ValueError('views must contain unique values from B, C, D, E, F')
        stages = {stages} if isinstance(stages, str) else (set(stages) if stages is not None else None)
        designs = {designs} if isinstance(designs, str) else (set(designs) if designs is not None else None)
        if stages is not None and stages - {'floorplan', 'placement', 'cts', 'route'}:
            raise ValueError('Unknown stage')
        with (self.root / 'manifest.csv').open(newline='') as stream:
            self.rows = [r for r in csv.DictReader(stream)
                         if (stages is None or r['stage'] in stages)
                         and (designs is None or r['design'] in designs)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        return {**{k: row[k] for k in ('sample_id', 'design', 'version', 'stage')},
                'views': {v: torch.load(self.root / row[v], map_location='cpu', weights_only=False)
                          for v in self.views}}


def collate_multiview(samples):
    """Batch each native graph view separately, retaining sample identities."""
    from torch_geometric.data import Batch
    if not samples:
        raise ValueError('Empty batch')
    return {**{k: [s[k] for s in samples] for k in ('sample_id', 'design', 'version', 'stage')},
            'views': {v: Batch.from_data_list([s['views'][v] for s in samples])
                      for v in samples[0]['views']}}
