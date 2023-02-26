import json
from pathlib import Path

import numpy as np
import pandas as pd

from vercelrenshu.api import estimate_core
from vercelrenshu.db import IRTParameters
from vercelrenshu.resources import records_to_df


def test_estimate():
    irt = json.load(open(Path(__file__).parent / "irt_parameters.json"))
    irt_parameters = IRTParameters(
        a_df=records_to_df(irt[0]["a"], dtypes={"bmsmd5": str, "a": float}),
        b_df=records_to_df(irt[0]["b"], dtypes={"bmsmd5": str, "grade": int, "b": float}),
        b_average=np.array(irt[0]["b_average"]),
    )
    request_df = pd.DataFrame(
        [
            {"bmsmd5": "25424c3d4c970f8d975a91bcb513960d", "grade": 3},
            {"bmsmd5": "e14c46af34be0165a72e597e217e6f73", "grade": 4},
        ],
        columns=["bmsmd5", "grade"],
    )

    ret = estimate_core(request_df, irt_parameters)
    pass
