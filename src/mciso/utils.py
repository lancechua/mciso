import itertools as it

import numpy as np


def array_to_dok(X: np.ndarray) -> dict:
    """Convert numpy array to dictionary of keys"""
    if X.ndim > 1:
        return dict(zip(it.product(*[range(1, i + 1) for i in X.shape]), X.ravel()))
    else:
        return {idx + 1: val for idx, val in enumerate(X)}


def dok_to_array(dok: dict, index_offset: int = -1) -> np.ndarray:
    """Convert dictionary of keys to numpy array"""
    dok_adj = {
        (
            tuple((k_i + index_offset) for k_i in k)
            if isinstance(k, tuple)
            else ((k + index_offset),)
        ): v
        for k, v in dok.items()
    }
    shape = tuple(np.array(list(dok_adj.keys())).max(axis=0) - index_offset)
    res = np.full(shape, np.nan)
    res[tuple(zip(*dok_adj.keys()))] = list(dok_adj.values())
    return res
