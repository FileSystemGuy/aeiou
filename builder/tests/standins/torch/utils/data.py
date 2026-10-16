"""TEST FAKE of `torch.utils.data` (see `standins/standin.py`): a loader that takes the dataset's
items in the calling process, batch by batch, as many batches as the real one dispatches."""
import random

from standin import record


class Dataset:
    pass


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers, drop_last):
        record("DataLoader", batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
               samples=len(dataset))
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(len(order)).shuffle(order)
        n = len(order) // self.batch_size if self.drop_last else -(-len(order) // self.batch_size)
        for b in range(n):
            items = [self.dataset[i] for i in order[b * self.batch_size:(b + 1) * self.batch_size]]
            yield tuple(zip(*items)) if isinstance(items[0], tuple) else items
