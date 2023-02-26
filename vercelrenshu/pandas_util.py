from typing import Any

import pandas as pd


def records_to_df(
    records: list[dict[str, Any]],
    dtypes: dict[str, type],
):
    df = pd.DataFrame(records)

    if set(dtypes.keys()) != set(df.columns):
        raise ValueError(f"expected fields are: {dtypes.keys()}, actual: {df.columns}")

    missing = df.isna()
    if missing.any(axis=None):
        incomplete_row = df[missing.any(axis=1)].iloc[0]  # 1つでもフィールドに欠けがある行のうち、先頭のもの
        raise ValueError(f"some fields are missing: {incomplete_row.to_dict()}")

    return df.astype(dtypes)
