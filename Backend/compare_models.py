import numpy as np
import tensorflow as tf
from pathlib import Path

folder = Path(__file__).resolve().parent

original = tf.keras.models.load_model(
    folder / "efficientnetb0_cardd_final.keras",
    compile=False
)

converted = tf.keras.models.load_model(
    folder / "efficientnetb0_cardd_render.keras",
    compile=False
)

# Same test input for both models
rng = np.random.default_rng(42)
sample = rng.uniform(
    0, 255, size=(1, 224, 224, 3)
).astype("float32")

original_pred = original.predict(sample, verbose=0)
converted_pred = converted.predict(sample, verbose=0)

print("Original:", original_pred)
print("Converted:", converted_pred)

difference = np.max(
    np.abs(original_pred - converted_pred)
)

print("Maximum difference:", difference)

if np.allclose(original_pred, converted_pred, atol=1e-5):
    print("SUCCESS: Model predictions match!")
else:
    print("WARNING: Model predictions differ!")