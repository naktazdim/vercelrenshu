"""ここの関数はすべてscipyにあるものだが、ファイルサイズ縮小のため自力で実装してscipyのインストールを回避"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def expit(x: np.ndarray) -> np.ndarray:
    """scipy.special.expit(x)"""
    return 1.0 / (1.0 + np.exp(-x))


def minimize(
    f: Callable[[float], float],
    bounds: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1.0e-6,
    maxiter: int = 100,
) -> float:
    """scipy.optimize.minimize_scalar(f).x"""

    # Golden-section search
    # https://en.wikipedia.org/wiki/Golden-section_search

    golden_ratio = (math.sqrt(5) + 1.0) / 2.0
    l1, r1 = bounds
    for _ in range(maxiter):
        l2 = r1 - (r1 - l1) / golden_ratio
        r2 = l1 + (r1 - l1) / golden_ratio
        # l1 < l2 < r2 < r1

        if f(l2) < f(r2):
            r1 = r2
        else:
            l1 = l2

        if r1 - l1 < tol:
            break

    return l1


def interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """scipy.interpolate.interp1d(xp, fp, fill_value="extrapolate")(x)"""

    if x < xp[0]:  # extrapolate
        return (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x - xp[0]) + fp[0]
    elif xp[-1] < x:  # extrapolate
        return (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x - xp[-1]) + fp[-1]
    else:  # interpolate
        return np.interp(x, xp, fp)  # type: ignore
