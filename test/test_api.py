import pandas as pd

from vercelrenshu.api import estimate_core


def test_estimate():
    request_df = pd.DataFrame(
        [
            {"bmsmd5": "25424c3d4c970f8d975a91bcb513960d", "grade": 3},
            {"bmsmd5": "e14c46af34be0165a72e597e217e6f73", "grade": 4},
        ],
        columns=["bmsmd5", "grade"],
    )
    ret = estimate_core(request_df)
    pass
