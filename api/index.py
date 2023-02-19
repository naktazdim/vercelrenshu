import os
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from flask import Flask
from pymongo import MongoClient

app = Flask(__name__)

mongo_counter = 0
vercel_counter = 0


@dataclass
class IRTParameters:
    a: pd.DataFrame
    b: pd.DataFrame
    b_average: pd.DataFrame


@cache
def load_irt_data() -> IRTParameters:
    global mongo_counter
    client = MongoClient(os.environ["MONGODB_URI"])
    db = client["walkureTestDB"]
    irt = db["irt"].find_one()
    mongo_counter += 1
    return IRTParameters(
        pd.DataFrame(irt["a"]),
        pd.DataFrame(irt["b"]),
        pd.DataFrame(irt["b_average"]),
    )


@app.route("/")
def home():
    global vercel_counter
    vercel_counter += 1
    return str(load_irt_data())


@app.route("/counter")
def about():
    return f"vercel: {vercel_counter}, mongo: {mongo_counter}"
