import json
import uuid
from datetime import datetime
from time import sleep

import pyarrow.parquet as pq
import requests

table = pq.read_table("penguins.parquet")
data = table.to_pylist()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", "w") as f_target:
    for row in data:
        row["rowid"] = str(uuid.uuid4())
        sex = row["sex"]
        f_target.write(f"{row['rowid']},{sex}\n")
        resp = requests.post(
            "http://127.0.0.1:6000/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(row, cls=DateTimeEncoder),
        ).json()
        print(f"prediction: {resp['sex']}")
        sleep(1)
