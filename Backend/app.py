from pathlib import Path
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BASE_DIR.parent / "Frontend") if (BASE_DIR.parent / "Frontend").exists() else (BASE_DIR.parent / "frontend")

# Configure Keras backend & cache dirs
os.environ.setdefault("KERAS_BACKEND", "torch")
keras_home = BASE_DIR.parent / ".keras"
os.environ.setdefault("KERAS_HOME", str(keras_home))
os.environ.setdefault("MPLCONFIGDIR", str(keras_home / "mpl"))

import keras
import torch

MODEL_PATH = BASE_DIR / "efficientnetb0_cardd_keras3.keras"
SETTINGS_PATH = BASE_DIR / "settings.json"

app = Flask(__name__)
# Generous 50 MB limit to support high-resolution mobile photos
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
CORS(app)


# Custom error handlers to guarantee all errors are returned as JSON, never HTML
@app.errorhandler(413)
def handle_413_payload_too_large(error):
    return jsonify({
        "error": "The uploaded photo is too large (maximum 50 MB). Please choose a smaller photo."
    }), 413


@app.errorhandler(400)
def handle_400_bad_request(error):
    msg = getattr(error, "description", "Bad request. Please verify the uploaded image.")
    return jsonify({"error": msg}), 400


@app.errorhandler(404)
def handle_404_not_found(error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(405)
def handle_405_method_not_allowed(error):
    return jsonify({"error": "HTTP Method not allowed on this endpoint."}), 405


@app.errorhandler(500)
def handle_500_server_error(error):
    return jsonify({
        "error": "Internal server error occurred while processing the inspection. Please try again."
    }), 500


@app.after_request
def add_cors_and_cache_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    # Prevent aggressive caching of frontend HTML/JS/CSS in browsers
    if request.path in ("/", "/index.html", "/script.js", "/style.css"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)

CLASS_NAMES = settings["classes"]
THRESHOLDS = settings["thresholds"]

print("Loading EfficientNetB0 CarDD damage detection model...")
model = keras.models.load_model(str(MODEL_PATH), compile=False)

print("MODEL LOADED SUCCESSFULLY")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".tiff", ".tif", ".gif", ".heic", ".heif"
}


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "VehDetect AI backend is running",
        "model_loaded": True,
        "classes": CLASS_NAMES
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "No file selected. Please choose a photo."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"File extension '{suffix}' is not supported. Please upload a JPG, PNG, WebP, or JFIF image."
        }), 400

    try:
        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        # Attempt decoding with PIL
        try:
            image = Image.open(io.BytesIO(raw_bytes))
        except Exception:
            return jsonify({
                "error": "Could not decode image file. Please ensure it is a valid JPG, PNG, or WebP photo."
            }), 400

        # Correct EXIF orientation and convert to standard 3-channel RGB
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Match EfficientNetB0 CarDD dimensions (224 x 224)
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

        # Keep pixel values in 0-255 range as float32
        image_array = np.asarray(image, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)

        with torch.no_grad():
            raw_scores = model({"image": image_batch}, training=False).detach().cpu().numpy()[0]
        scores = [float(x) for x in raw_scores]

        predictions = []
        for name, score in zip(CLASS_NAMES, scores):
            threshold = float(THRESHOLDS[name])
            predictions.append({
                "name": name,
                "confidence": score,
                "threshold": threshold,
                "detected": score >= threshold
            })

        return jsonify({
            "message": "Prediction completed successfully",
            "scores": scores,
            "predictions": predictions
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": f"Server encountered an error processing this image: {str(error)}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting VehDetect server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
