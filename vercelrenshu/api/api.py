from typing import Union

import pandas as pd
from fastapi import FastAPI

from vercelrenshu.api.logic import estimate, theta_to_hoshi
from vercelrenshu.api.request_body import RequestLR2, RequestMD5, lr2_to_lamps_df, md5_to_lamps_df

app = FastAPI()


@app.get("/")
def home():
    return "home"


@app.post("/estimate_md5")
def estimate_md5(request: RequestMD5) -> str:
    lamps_df = md5_to_lamps_df(request)
    return _estimate(lamps_df)


@app.post("/estimate_lr2")
def estimate_lr2(request: RequestLR2) -> str:
    lamps_df = lr2_to_lamps_df(request)
    return _estimate(lamps_df)


def _estimate(lamps_df: pd.DataFrame) -> str:
    estimation = estimate(lamps_df)
    hoshi = theta_to_hoshi(estimation.theta)
    print(estimation.recommendation)
    print(estimation.reverse_recommendation)
    return f"★{hoshi:.02f}"
