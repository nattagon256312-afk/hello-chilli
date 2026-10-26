from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("predict.html")

model = load_model("chilli_model.keras", custom_objects={"preprocess_input": preprocess_input},
                   safe_mode=False,
                   )

class_names = [
    "แมลงหวี่ขาวในพริก",
    "ใบเหลือง",
    "ปกติ",
    "โรคกุ้งแห้ง",
    "โรคโคนเน่า",
    "โรคใบหงิกจากไวรัส",
    "โรคใบจุด",
    "โรคเส้นใบด่างประ"
    ]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "ไม่พบภาพ"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert("RGB")
        img = img.resize((160, 160))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predicttions = model(img_array, training=False).numpy()
        index = int(np.argmax(predicttions[0]))

        return jsonify({
            "class": class_names[index],
            "confidence": float(predicttions[0][index])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)