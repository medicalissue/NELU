""" CIFAR-100 N-grid data loader for the GEM benchmark.

Mirrors ``gem.datasets.cifair.ciFAIR10`` exactly (subclass the
torchvision dataset, then carve disjoint per-class index slices for
each split), but for CIFAR-100 and with the *number of training images
per class N* parameterised in the split name. This gives the
data-size-axis the small-data paper needs (Brigato Protocol-2 grid),
while keeping GEM's split/HPO/eval machinery untouched.

Split-name grammar (parsed by ``__init__``):
    'train{i}_n{N}'     - N images/class for split i (HPO train set)
    'val{i}'            - fixed 20 images/class for split i (HPO val)
    'trainval{i}_n{N}'  - train{i}_n{N} + val{i}  (final train set)
    'fulltrain'         - full CIFAR-100 train set (50,000 imgs)
    'test0'             - the official CIFAR-100 test set (10,000 imgs)

i in {0,1,2}; the three splits use disjoint per-class index ranges so
results can be averaged over splits exactly like ciFAIR-10. N ranges
over the Brigato grid (e.g. 10,20,40,80,160,320,640); CIFAR-100 has
500 train imgs/class so N<=480 leaves room for the 20/class val block.

The val block is a FIXED 20 imgs/class right after the largest N we
use, so val never overlaps train regardless of N — keeping HPO honest
across the whole N-grid (same val set for every N at a given split).
"""

import numpy as np
import torchvision.datasets

# Per-class index layout (CIFAR-100 has 500 train imgs/class):
#   split 0: train uses [0 : N],            val uses [480 : 500]
#   split 1: train uses [0 : N] of a rolled order ... -> we instead use
#            disjoint blocks: split i train starts at i*BLOCK.
# Keep it simple and disjoint like ciFAIR10: split i uses class indices
# [i*160 : i*160 + N] for train and [i*160 + 140 : i*160 + 160] for val
# (160-wide block per split, 3 splits -> 480 <= 500). Max N = 140.
_BLOCK = 160
_VAL_PER_CLASS = 20
_MAX_N = _BLOCK - _VAL_PER_CLASS  # 140


class CIFAR100Ngrid(torchvision.datasets.CIFAR100):
    """ CIFAR-100 with N-parameterised small-data splits. """

    @staticmethod
    def get_ds_name():
        return "cifar100"

    def __init__(self, root, split, transform=None, target_transform=None,
                 download=True):
        super().__init__(
            root,
            train=(split != "test0"),
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if split in ("fulltrain", "test0"):
            return  # use the full set as loaded by torchvision

        # Parse "<kind>{i}[_n{N}]"
        kind, i, n = self._parse_split(split)
        if not (0 <= i <= 2):
            raise ValueError(f"split id must be 0..2, got {i} (split={split!r})")
        base = i * _BLOCK

        class_members = {c: [] for c in range(len(self.classes))}
        for idx, lbl in enumerate(self.targets):
            class_members[lbl].append(idx)

        def take(start, end):
            return np.concatenate(
                [mem[start:end] for mem in class_members.values()]
            )

        if kind == "train":
            if n is None or n < 1 or n > _MAX_N:
                raise ValueError(
                    f"train split needs 1<=N<={_MAX_N}, got N={n}"
                )
            indices = take(base, base + n)
        elif kind == "val":
            indices = take(base + _MAX_N, base + _BLOCK)  # fixed 20/class
        elif kind == "trainval":
            if n is None or n < 1 or n > _MAX_N:
                raise ValueError(
                    f"trainval split needs 1<=N<={_MAX_N}, got N={n}"
                )
            tr = take(base, base + n)
            va = take(base + _MAX_N, base + _BLOCK)
            indices = np.concatenate([tr, va])
        else:
            raise ValueError(f"unknown split kind {kind!r} in {split!r}")

        self.data = self.data[indices]
        self.targets = np.asarray(self.targets)[indices]

    @staticmethod
    def _parse_split(split: str):
        """'train0_n40' -> ('train', 0, 40); 'val1' -> ('val', 1, None)."""
        n = None
        if "_n" in split:
            split, n_str = split.split("_n")
            n = int(n_str)
        for kind in ("trainval", "train", "val"):
            if split.startswith(kind):
                return kind, int(split[len(kind):]), n
        raise ValueError(f"cannot parse split name {split!r}")

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_input_channels(self):
        return 3
