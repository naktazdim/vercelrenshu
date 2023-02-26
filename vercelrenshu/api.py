import numpy as np
import pandas as pd
from flask import Flask, request

from vercelrenshu.calc import expit, interp, minimize
from vercelrenshu.pandas_util import records_to_df
from vercelrenshu.resources import BMSTypeCategory, load_bms_meta, load_irt_parameters

app = Flask(__name__)


def make_lamps_df_lr2(lamps_dict: list[dict]) -> pd.DataFrame:
    lamps_df_lr2 = records_to_df(lamps_dict, {"type": BMSTypeCategory, "lr2_id": int, "grade": int})
    bms_meta_df = load_bms_meta()[["type", "lr2_id", "bmsmd5"]]
    lamps_df = pd.merge(lamps_df_lr2, bms_meta_df, on=["type", "lr2_id"], how="inner")  # bmsmd5を付与
    return lamps_df[["bmsmd5", "grade"]]  # 不要なカラムを削除して返す


def make_lamps_df_md5(lamps_dict: list[dict]) -> pd.DataFrame:
    return records_to_df(lamps_dict, {"bmsmd5": str, "grade": int})


def estimate_core(lamps_df: pd.DataFrame) -> pd.DataFrame:
    irt_parameters = load_irt_parameters()

    a = pd.merge(lamps_df, irt_parameters.a_df, on="bmsmd5", how="inner")["a"].to_numpy()
    b_lower = pd.merge(lamps_df, irt_parameters.b_df, on=["bmsmd5", "grade"], how="inner")["b"].to_numpy()
    lamps_df["grade"] += 1
    b_upper = pd.merge(lamps_df, irt_parameters.b_df, on=["bmsmd5", "grade"], how="inner")["b"].to_numpy()

    def target(_t: float) -> float:
        # negative log likelihood to minimize
        return -np.log(expit(a * (_t - b_lower)) - expit(a * (_t - b_upper))).sum()

    t = minimize(target)

    b_average_df = irt_parameters.b_average_df
    hoshi = interp(t, b_average_df["b"].to_numpy(), b_average_df["hoshi"].to_numpy())

    return hoshi


@app.route("/")
def home():
    return "home"


@app.route("/estimate", methods=["POST"])
def estimate():
    request_dict = request.get_json()
    lamps_df = make_lamps_df_md5(request_dict)
    hoshi = estimate_core(lamps_df)
    return str(hoshi)


@app.route("/estimate_lr2", methods=["POST"])
def estimate_lr2():
    request_dict = request.get_json()
    lamps_df = make_lamps_df_lr2(request_dict)
    hoshi = estimate_core(lamps_df)
    return str(hoshi)
