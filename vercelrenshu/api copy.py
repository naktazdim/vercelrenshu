import numpy as np
import pandas as pd
from flask import Flask, request

from vercelrenshu.calc import expit, interp, minimize
from vercelrenshu.pandas_util import records_to_df
from vercelrenshu.resources import BMSTypeCategory, load_bms_meta, load_irt_parameters

app = Flask(__name__)


def make_lamps_df(lamps_dict: list[dict], id_type: str) -> pd.DataFrame:
    if id_type == "lr2":
        lamps_df_lr2 = records_to_df(lamps_dict, {"type": BMSTypeCategory, "lr2_id": int, "grade": int})
        bms_meta_df = load_bms_meta()[["type", "lr2_id", "bmsmd5"]]
        lamps_df = pd.merge(lamps_df_lr2, bms_meta_df, on=["type", "lr2_id"], how="inner")  # bmsmd5を付与
        return lamps_df[["bmsmd5", "grade"]]  # 不要なカラムを削除して返す
    elif id_type == "bmsmd5":
        return records_to_df(lamps_dict, {"bmsmd5": str, "grade": int})
    else:
        raise ValueError("field 'id_type' must be 'lr2' or 'bmsmd5'")


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


@app.route("/")
def home():
    return "home"


@app.route("/estimate", methods=["POST"])
def estimate():
    request_dict = request.get_json()
    lamps_df = make_lamps_df(request_dict["lamps"], request_dict["id_type"])
    t = estimate_core(lamps_df)
    hoshi = to_hoshi(t)
    return str(hoshi)
