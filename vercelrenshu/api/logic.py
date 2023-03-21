from dataclasses import dataclass

import numpy as np
import pandas as pd

from vercelrenshu.resources import a_df, b_average_df, b_df
from vercelrenshu.util.calc import expit, interp, minimize


@dataclass
class Estimation:
    theta: float
    recommendation: pd.DataFrame
    reverse_recommendation: pd.DataFrame


def estimate(lamps_df: pd.DataFrame) -> Estimation:
    theta = _estimate_theta(lamps_df)
    probabilities_df = _calculate_all_probabilities(lamps_df, theta)
    recommendation = _recommendation(probabilities_df, threshold=0.2)
    reverse_recommendation = _reverse_recommendation(probabilities_df, threshold=0.5)
    return Estimation(theta, recommendation, reverse_recommendation)


def theta_to_hoshi(theta: float) -> float:
    return interp(
        theta,
        b_average_df()["b"].to_numpy(),
        b_average_df()["hoshi"].to_numpy(),
    )


# -------------------------------- #


def _a(bmsmd5: pd.Series) -> np.ndarray:
    return pd.merge(
        bmsmd5,
        a_df(),
        on="bmsmd5",
        how="inner",
    )["a"].to_numpy()


def _b(bmsmd5: pd.Series, grade: pd.Series) -> np.ndarray:
    return pd.merge(
        pd.concat([bmsmd5, grade], axis=1),
        b_df(),
        on=["bmsmd5", "grade"],
        how="inner",
    )["b"].to_numpy()


def _prob(theta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return expit(a * (theta - b))


def _estimate_theta(lamps_df: pd.DataFrame) -> float:
    a = _a(lamps_df["bmsmd5"])
    b = _b(lamps_df["bmsmd5"], lamps_df["grade"])
    b_upper = _b(lamps_df["bmsmd5"], lamps_df["grade"] + 1)

    def _negative_log_likelihood(theta: float):
        likelihood = _prob(theta, a, b) - _prob(theta, a, b_upper)
        return -np.log(likelihood).sum()

    return minimize(_negative_log_likelihood)


def _calculate_all_probabilities(lamps_df: pd.DataFrame, theta: float) -> pd.DataFrame:
    a_b_df = pd.merge(
        a_df(),
        b_df().query("2 <= grade & grade <= 5"),
        on="bmsmd5",
    )
    probabilities_df = (
        a_b_df
        .assign(probability=_prob(theta, a_b_df["a"], a_b_df["b"]))
        [["bmsmd5", "grade", "probability"]]
    )  # fmt: skip
    return (
        pd.merge(
            probabilities_df.rename(columns={"grade": "target_grade"}),
            lamps_df.rename(columns={"grade": "current_grade"}),
            on="bmsmd5",
            how="left",
        ).fillna(0, downcast="infer")
        [["bmsmd5", "current_grade", "target_grade", "probability"]]
    )  # fmt: skip


def _recommendation(probabilities_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return (
        probabilities_df
        .query(f"probability >= {threshold} &"
               " target_grade > current_grade")
        .sort_values(by="probability", ascending=False)
    )  # fmt: skip


def _reverse_recommendation(probabilities_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return (
        probabilities_df
        .query(f"probability <= {threshold} &"
               " target_grade == current_grade")
        .drop("target_grade", axis=1)
        .sort_values(by="probability", ascending=True)
    )  # fmt: skip
