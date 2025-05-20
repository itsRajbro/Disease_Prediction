# scripts/train_ocular.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import convnext
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# Configuration
DATA_DIR = "combined_dataset_ocular"  # Path to ocular dataset
IMG_SIZE = (224, 224)  # Standard size for fundus images
BATCH_SIZE = 32
EPOCHS = 40  # Increased epochs for better convergence
CLASS_NAMES = ["A", "C", "D", "G", "H", "M", "N", "O"]
MODEL_PATH = "model/ocular_model.keras"  # Using modern .keras format

# Create model directory
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased from 20
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    brightness_range=[0.7, 1.3],  # Wider brightness range
    channel_shift_range=50.0,  # Added channel shifts
    fill_mode='wrap'
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

# ConvNeXt Architecture (State-of-the-art for medical imaging)
def create_convnext_model():
    base_model = convnext.ConvNeXtTiny(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        weights='imagenet',
        include_top=False
    )
    
    # Freeze initial layers
    for layer in base_model.layers[:150]:
        layer.trainable = False
        
    # Custom Head with Attention-like Features
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)

model = create_convnext_model()

# Advanced Learning Rate Schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Compile with Precision/Recall Tracking
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Enhanced Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
]

# Initial Training Phase
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tuning Phase (Unfreeze more layers)
print("\nStarting fine-tuning...")
for layer in model.layers[100:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')]
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=15,
    callbacks=callbacks
)

# Final Save
model.save(MODEL_PATH)
print(f"Ocular disease model saved to {MODEL_PATH}")

# Comprehensive Evaluation
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

results = model.evaluate(test_generator)
print("\nTest Results:")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")
