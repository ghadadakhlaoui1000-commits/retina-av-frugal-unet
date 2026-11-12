
import numpy as np
IGNORE = 255

def normalize(img01):
    """img01 : HxWx3 float [0,1] -> 3xHxW standardisé par canal (statistiques calculées dans le champ de vue)."""
    fov = img01.max(axis=2) > 0.04
    px = img01[fov] if fov.sum() > 100 else img01.reshape(-1, 3)
    mean, std = px.mean(0), px.std(0) + 1e-6
    return np.ascontiguousarray(((img01 - mean) / std).transpose(2, 0, 1), dtype=np.float32)

class PatchSampler:
    """Patches d'entraînement : centrés sur un vaisseau avec proba p_vessel, + augmentation."""
    def __init__(self, images, labels, indices, size=256, length=1000, p_vessel=0.7, seed=None):
        self.images, self.labels, self.idx = images, labels, list(indices)
        self.size, self.length, self.p_vessel, self.seed = size, length, p_vessel, seed
        r = np.random.default_rng(0)
        self.vessel = {}
        for i in self.idx:
            m = np.argwhere((np.asarray(labels[i]) == 1) | (np.asarray(labels[i]) == 2))
            if len(m) > 20000:
                m = m[r.choice(len(m), 20000, replace=False)]
            self.vessel[i] = m
    def __len__(self):
        return self.length
    def _make(self, k):
        rng = np.random.default_rng() if self.seed is None else np.random.default_rng([self.seed, k])
        i = self.idx[int(rng.integers(len(self.idx)))]
        H, W = self.labels[i].shape; s = self.size
        if rng.random() < self.p_vessel and len(self.vessel[i]):
            cy, cx = self.vessel[i][int(rng.integers(len(self.vessel[i])))]
            j = s // 4
            y0 = int(np.clip(cy - s // 2 + rng.integers(-j, j + 1), 0, H - s))
            x0 = int(np.clip(cx - s // 2 + rng.integers(-j, j + 1), 0, W - s))
        else:
            y0, x0 = int(rng.integers(0, H - s + 1)), int(rng.integers(0, W - s + 1))
        img = np.asarray(self.images[i][y0:y0 + s, x0:x0 + s]).astype(np.float32) / 255.0
        lab = np.asarray(self.labels[i][y0:y0 + s, x0:x0 + s]).copy()
        if rng.random() < 0.5: img, lab = img[:, ::-1], lab[:, ::-1]
        if rng.random() < 0.5: img, lab = img[::-1], lab[::-1]
        r90 = int(rng.integers(4)); img, lab = np.rot90(img, r90), np.rot90(lab, r90)
        img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)                 # gamma
        img = (img - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1)   # contraste / luminosité
        img = np.clip(img * rng.uniform(0.9, 1.1, size=3), 0, 1)           # gain par canal
        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(lab)
    def get(self, k):
        img, lab = self._make(k)
        return normalize(img), lab.astype(np.int64)
    def get_raw(self, k):
        img, lab = self._make(k)
        return (img * 255).astype(np.uint8), lab

class FullSampler:
    """Images entières (validation / test), sans augmentation."""
    def __init__(self, images, labels, indices):
        self.images, self.labels, self.idx = images, labels, list(indices)
    def __len__(self):
        return len(self.idx)
    def get(self, k):
        i = self.idx[k]
        img = np.asarray(self.images[i]).astype(np.float32) / 255.0
        return normalize(img), np.asarray(self.labels[i]).astype(np.int64)

try:
    import torch
    from torch.utils.data import Dataset
    class RetinaPatches(PatchSampler, Dataset):
        def __getitem__(self, k):
            x, y = self.get(k); return torch.from_numpy(x), torch.from_numpy(y)
    class RetinaFull(FullSampler, Dataset):
        def __getitem__(self, k):
            x, y = self.get(k); return torch.from_numpy(x), torch.from_numpy(y)
except ImportError:
    pass
