import os
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from flask import Flask
from pymongo import MongoClient

app = Flask(__name__)

counter = 0


@dataclass
class IRTParameters:
    a: pd.DataFrame
    b: pd.DataFrame
    b_average: pd.DataFrame


@cache
def load_irt_data() -> IRTParameters:
    global counter
    client = MongoClient(os.environ["MONGODB_URI"])
    db = client["walkureTestDB"]
    irt = db["irt"].find_one()
    counter += 1
    return IRTParameters(
        pd.DataFrame(irt["a"]),
        pd.DataFrame(irt["b"]),
        pd.DataFrame(irt["b_average"]),
    )


@app.route("/")
def home():
    return str(load_irt_data())


@app.route("/counter")
def about():
    return str(counter)
