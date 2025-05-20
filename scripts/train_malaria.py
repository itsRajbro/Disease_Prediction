import tensorflow as tf
from tensorflow.keras.applications import convnext
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# Configuration
DATA_DIR = "combined_dataset_malaria"  # Update this path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
CLASS_NAMES = ["Parasitized", "Uninfected"]
MODEL_PATH = "model/malaria_model.keras"

# Create model directory
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Advanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Data Generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ConvNeXt Architecture
def create_convnext_model():
    base_model = convnext.ConvNeXtTiny(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        weights='imagenet',
        include_top=False
    )
    
    # Freeze initial layers
    for layer in base_model.layers[:150]:
        layer.trainable = False
        
    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)

model = create_convnext_model()

# Learning Rate Schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96
)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True),
]

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tuning Phase
print("\nStarting fine-tuning...")
for layer in model.layers[100:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=10,
    callbacks=callbacks
)

# Save Final Model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Evaluation
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy, precision, recall = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
