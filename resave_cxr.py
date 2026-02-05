import tensorflow as tf

model = tf.keras.models.load_model("model/cxr_model.h5", compile=False)
model.save("model/cxr_model.keras")

print("✅ CXR model re-saved as model/cxr_model.keras")
