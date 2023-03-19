from typing import Any

import pandas as pd


def records_to_df(
    records: list[dict[str, Any]],
    dtypes: dict[str, type],
):
    return pd.DataFrame(records).astype(dtypes)
