import tensorflow as tf

model = tf.keras.models.load_model(
    "efficientnetb0_cardd_final.keras",
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)