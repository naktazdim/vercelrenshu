from typing import Literal

import pandas as pd
from pydantic import BaseModel, conint, constr

from vercelrenshu.db import bms_meta_df


class LampLR2(BaseModel):
    type: Literal["bms", "course"]
    lr2_id: int
    grade: conint(ge=1, le=5)  # type: ignore


class RequestLR2(BaseModel):
    __root__: list[LampLR2]


class LampMD5(BaseModel):
    bmsmd5: constr(regex=r"^([0-9a-f]{32}|[0-9a-f]{160})$")  # type: ignore  # NOQA
    grade: conint(ge=1, le=5)  # type: ignore


class RequestMD5(BaseModel):
    __root__: list[LampMD5]


def lr2_to_lamps_df(request: RequestLR2) -> pd.DataFrame:
    lr2_df = pd.DataFrame(request.dict()["__root__"]).astype({"type": str, "lr2_id": int, "grade": int})
    return pd.merge(
        lr2_df,
        bms_meta_df(),
        on=["type", "lr2_id"],
        how="inner",
    )[["bmsmd5", "grade"]]


def md5_to_lamps_df(request: RequestMD5) -> pd.DataFrame:
    return pd.DataFrame(request.dict()["__root__"]).astype({"bmsmd5": str, "grade": int})
