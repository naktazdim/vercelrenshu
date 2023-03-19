from typing import Literal, Union

import pandas as pd
from pydantic import BaseModel, conint, constr

from vercelrenshu.resources import BMSTypeCategory, load_bms_meta
from vercelrenshu.util.df import records_to_df


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
    lr2_df = records_to_df(
        request.dict()["__root__"],
        dtypes={"type": BMSTypeCategory, "lr2_id": int, "grade": int},
    )
    return pd.merge(
        lr2_df,
        load_bms_meta(),
        on=["type", "lr2_id"],
        how="inner",
    )[["bmsmd5", "grade"]]


def md5_to_lamps_df(request: RequestMD5) -> pd.DataFrame:
    return records_to_df(
        request.dict()["__root__"],
        dtypes={"bmsmd5": str, "grade": int},
    )
