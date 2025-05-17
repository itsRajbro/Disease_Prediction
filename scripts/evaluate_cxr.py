import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = r"C:\Users\rajay\disease_predictor\disease_predictor\model\cxr_model.h5"
DATA_DIR = r"C:\Users\rajay\disease_predictor\disease_predictor\combined_dataset_cxr"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    f"{DATA_DIR}/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

loss, acc = model.evaluate(test_generator)
print(f"Test accuracy: {acc*100:.2f}%")

y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
