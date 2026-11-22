import tensorflow as tf
from pathlib import Path

base_dir = Path(__file__).resolve().parent

original = base_dir / "efficientnetb0_cardd_final.keras"
output = base_dir / "efficientnetb0_cardd_render.keras"

print("Loading original model...")

model = tf.keras.models.load_model(
    str(original),
    compile=False
)

print("Saving Render-compatible copy...")

model.save(str(output))

print("MODEL SAVED SUCCESSFULLY")
print("Input:", model.input_shape)
print("Output:", model.output_shape)