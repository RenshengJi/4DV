"""
EasyDataset - Support for dataset multiplication and resizing operators.
Migrated from dust3r for compatibility with config syntax.
"""
import numpy as np
from dust3r.datasets.base.batched_sampler import (
    BatchedRandomSampler,
    CustomRandomSampler,
)


class EasyDataset:
    """
    A dataset that supports easy resizing and combination using operators.

    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x
        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)
        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other):
        """Concatenate two datasets"""
        return CatDataset([self, other])

    def __rmul__(self, factor):
        """Multiply dataset size by factor (e.g., 2 * dataset)"""
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        """Resize dataset to specific size (e.g., 1000 @ dataset)"""
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        """Set epoch for dataset (used by ResizedDataset for shuffling)"""
        pass  # nothing to do by default

    def make_sampler(
        self, batch_size, shuffle=True, drop_last=True, world_size=1, rank=0, fixed_length=False
    ):
        """
        Create a simple random sampler for this dataset.
        No longer uses tuple indices - just returns scene indices.
        """
        if not shuffle:
            raise NotImplementedError("Non-shuffled sampling not yet supported")

        # Use standard PyTorch RandomSampler instead of CustomRandomSampler
        # since we no longer need tuple indices (scene_idx, ar_idx, nview)
        from torch.utils.data import RandomSampler, BatchSampler

        sampler = RandomSampler(self)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
        return batch_sampler


class MulDataset(EasyDataset):
    """Artificially augment dataset size by repeating it multiple times."""

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        # Simply map to base dataset with divided index
        return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class ResizedDataset(EasyDataset):
    """Artificially change the size of a dataset to a specific value."""

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        return self.new_size

    def __repr__(self):
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        """Shuffle indices based on epoch"""
        # This random shuffle only depends on the epoch
        rng = np.random.default_rng(seed=epoch + 777)

        # Shuffle all indices
        perm = rng.permutation(len(self.dataset))

        # Rotary extension until target size is met
        shuffled_idxs = np.concatenate(
            [perm] * (1 + (len(self) - 1) // len(self.dataset))
        )
        self._idxs_mapping = shuffled_idxs[: self.new_size]

        assert len(self._idxs_mapping) == self.new_size

    def __getitem__(self, idx):
        assert hasattr(
            self, "_idxs_mapping"
        ), "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"

        # Simply map to base dataset with shuffled index
        return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        # Remove uselessly long transform descriptions
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            )
            for dataset in self.datasets
        )

    def set_epoch(self, epoch):
        """Set epoch for all concatenated datasets"""
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError()

        # Find which dataset this index belongs to
        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        return dataset[new_idx]

    @property
    def _resolutions(self):
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions)
        return resolutions

    @property
    def num_views(self):
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views
        return num_views
