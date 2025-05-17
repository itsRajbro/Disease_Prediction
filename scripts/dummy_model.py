import tensorflow as tf
from tensorflow.keras import layers

# Create a simple dummy model
model = tf.keras.Sequential([
    layers.Input(shape=(224, 224, 3)),  # Input size same as preprocess
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(2, activation='softmax')  # 2 classes (just for dummy)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save("model/dummy_model.h5")

print("✅ Dummy model created successfully!")
