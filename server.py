import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import base64
import io
import logging

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, make_response
from PIL import Image

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = "./aiy-tensorflow1-vision-classifier-food-v1-v1"
LABELS_FILE = "./labels.txt"
IMAGE_SIZE = 192
PORT = 5001

# ---------------------------------------------------------------------------
# Load labels
# ---------------------------------------------------------------------------
def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f.readlines()]
    return raw


def clean_label(raw: str) -> str:
    return raw.replace("_", " ").title()


labels: list[str] = load_labels(LABELS_FILE)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"[server] Loading model from {MODEL_DIR} …")
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures["default"]
print("[server] Model loaded successfully.")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/classify", methods=["OPTIONS"])
@app.route("/health", methods=["OPTIONS"])
def options_handler():
    return make_response("", 204)


def preprocess_image(b64_string: str) -> tf.Tensor:
    """Decode a base64 image string and return a (1, 192, 192, 3) float32 tensor."""
    # Strip data-URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    arr = np.array(image, dtype=np.float32) / 255.0          # shape (192, 192, 3)
    arr = np.expand_dims(arr, axis=0)                         # shape (1, 192, 192, 3)
    return tf.constant(arr, dtype=tf.float32)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_DIR, "labels": len(labels)})


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True, silent=True)

    if not data or "image" not in data:
        return jsonify({"success": False, "error": "Missing 'image' field in JSON body"}), 400

    try:
        tensor = preprocess_image(data["image"])
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image decode error: {exc}"}), 400

    try:
        output = infer(images=tensor)
        # The AIY food model returns a dict; the scores key is typically 'default'
        # but fall back gracefully to whichever key holds a (1, N) tensor.
        scores_tensor = None
        for key in ("default", "scores", "probabilities", "output"):
            if key in output:
                scores_tensor = output[key]
                break
        if scores_tensor is None:
            # Just grab the first value
            scores_tensor = next(iter(output.values()))

        scores = scores_tensor.numpy().flatten()          # shape (num_classes,)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Inference error: {exc}"}), 500

    # Top-5 predictions
    top5_indices = np.argsort(scores)[::-1][:5]
    predictions = []
    for idx in top5_indices:
        label_raw = labels[idx] if idx < len(labels) else f"class_{idx}"
        predictions.append(
            {
                "label": clean_label(label_raw),
                "confidence": round(float(scores[idx]), 4),
            }
        )

    return jsonify({"success": True, "predictions": predictions})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[server] Starting on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
