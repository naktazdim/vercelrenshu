from __future__ import annotations

import pickle
from functools import cache
from importlib.resources import files

import pandas as pd

import vercelrenshu.api


@cache
def _load_db():
    return pickle.load(files(vercelrenshu.api).joinpath("db.pickle").open("rb"))


def a_df() -> pd.DataFrame:
    return _load_db()["a"]


def b_df() -> pd.DataFrame:
    return _load_db()["b"]


def b_average_df() -> pd.DataFrame:
    return _load_db()["b_average"]


def bms_meta_df() -> pd.DataFrame:
    return _load_db()["bms_meta"]
