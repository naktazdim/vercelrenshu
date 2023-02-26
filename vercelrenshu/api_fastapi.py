import time
from typing import Literal, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from vercelrenshu.calc import expit, interp, minimize
from vercelrenshu.pandas_util import records_to_df
from vercelrenshu.resources import BMSTypeCategory, load_bms_meta, load_irt_parameters


class LampLR2(BaseModel):
    type: Literal["bms", "course"]
    lr2_id: int
    grade: int


class RequestLR2(BaseModel):
    __root__: list[LampLR2]


class LampMD5(BaseModel):
    bmsmd5: str
    grade: int


class RequestMD5(BaseModel):
    __root__: list[LampMD5]


app = FastAPI()


def make_lamps_df_lr2(request: RequestLR2) -> pd.DataFrame:
    lamps_df_lr2 = records_to_df(request.dict()["__root__"], {"type": BMSTypeCategory, "lr2_id": int, "grade": int})
    bms_meta_df = load_bms_meta()[["type", "lr2_id", "bmsmd5"]]
    lamps_df = pd.merge(lamps_df_lr2, bms_meta_df, on=["type", "lr2_id"], how="inner")  # bmsmd5を付与
    return lamps_df[["bmsmd5", "grade"]]  # 不要なカラムを削除して返す


def make_lamps_df_md5(request: RequestMD5) -> pd.DataFrame:
    return records_to_df(request.dict()["__root__"], {"bmsmd5": str, "grade": int})


def estimate_core(lamps_df: pd.DataFrame) -> pd.DataFrame:
    irt_parameters = load_irt_parameters()

    a = pd.merge(lamps_df, irt_parameters.a_df, on="bmsmd5", how="inner")["a"].to_numpy()
    b_lower = pd.merge(lamps_df, irt_parameters.b_df, on=["bmsmd5", "grade"], how="inner")["b"].to_numpy()
    lamps_df["grade"] += 1
    b_upper = pd.merge(lamps_df, irt_parameters.b_df, on=["bmsmd5", "grade"], how="inner")["b"].to_numpy()

    def target(_t: float) -> float:
        # negative log likelihood to minimize
        return -np.log(expit(a * (_t - b_lower)) - expit(a * (_t - b_upper))).sum()

    return minimize(target)


def to_hoshi(t: float) -> float:
    b_average_df = load_irt_parameters().b_average_df
    return interp(t, b_average_df["b"].to_numpy(), b_average_df["hoshi"].to_numpy())


@app.get("/")
def home():
    return "home"


@app.post("/estimate")
def estimate(request: Union[RequestLR2, RequestMD5]):
    start = time.time()
    if type(request) is RequestLR2:
        lamps_df = make_lamps_df_lr2(request)
    elif type(request) is RequestMD5:
        lamps_df = make_lamps_df_md5(request)
    t = estimate_core(lamps_df)
    hoshi = to_hoshi(t)
    t = time.time() - start
    return f"★{hoshi:.02f}, {t:.03f} sec."
