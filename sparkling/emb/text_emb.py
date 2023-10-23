from abc import ABC, abstractmethod

from .deep_emb import DeepEmb


class TextBatch:
    """ Lightweight custom implementation for batch text sampling """

    def __init__(self, series, batch_size):
        self.series, self.batch_size = series.tolist(), batch_size
        self.cur_idx, self.len = 0, len(series)

    def __iter__(self):
        return self

    def __next__(self):
        lower = self.cur_idx
        upper = min(lower + self.batch_size, self.len)
        if lower == upper:
            self.cur_idx = 0
            raise StopIteration
        self.cur_idx = upper
        return self.series[lower:upper]


class TextEmb(DeepEmb, ABC):
    """ Basic marker for text modality preprocessors """

    def __init__(self, name: str, model_name: str, orig_dim: int, reduce_dim: bool, **kwargs):
        super().__init__(name, model_name, orig_dim, reduce_dim, **kwargs)

    @staticmethod
    @abstractmethod
    def from_zoo(name: str, reduce_dim: bool, model, batch_size: int):
        pass
