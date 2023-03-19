from __future__ import annotations

import bz2
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
from pandas.api.types import CategoricalDtype

from vercelrenshu.util.df import records_to_df

_RESOURCE_DIR = Path(__file__).parent.parent / "resources"

BMSTypeCategory = CategoricalDtype(["bms", "course"])


@dataclass
class IRTParameters:
    a_df: pd.DataFrame
    b_df: pd.DataFrame
    b_average_df: pd.DataFrame


@cache
def load_irt_parameters() -> IRTParameters:
    d = json.load(bz2.open(_RESOURCE_DIR / "irt_parameters.json.bz2"))
    a_df = records_to_df(d["a"], dtypes={"bmsmd5": str, "a": float})
    b_df = records_to_df(d["b"], dtypes={"bmsmd5": str, "b": float, "grade": int})
    b_average_df = records_to_df(d["b_average"], dtypes={"b": float, "hoshi": int})
    return IRTParameters(a_df, b_df, b_average_df)


@cache
def load_bms_meta() -> pd.DataFrame:
    d = json.load(bz2.open(_RESOURCE_DIR / "bms_meta.json.bz2"))
    return records_to_df(d, dtypes={"bmsmd5": str, "type": BMSTypeCategory, "lr2_id": int, "title": str})
