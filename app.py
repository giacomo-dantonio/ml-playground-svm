import base64
import io
import joblib
import json
import re

from flask import Flask
from flask import request

import predict

app = Flask(__name__)

b64exp = re.compile("data:(?P<mime_type>.*);base64,(?P<base64>.*)$")
model = joblib.load("model_rbf_shifted.pkl")

@app.route("/")
def index_route():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict_digit():
    data = request.json
    digit = data.get("digit")
    if digit is not None:
        bytes = get_bytes_from_canvas(digit)

        image = predict.load_image(io.BytesIO(bytes))
        prediction = model.predict([image])

        return json.dumps({
            "prediction": int(prediction[0])
        })


def get_bytes_from_canvas(dataimg : str):
    match = b64exp.match(dataimg)
    if match is not None:
        encoded_str = match.group("base64")
        bytes = base64.b64decode(encoded_str)
        return bytes