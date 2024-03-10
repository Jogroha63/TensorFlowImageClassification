import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from datetime import date

# Setzen von Werte für das Datasets
img_height = 180
img_width = 180

# Erstellen der Datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
  "./food_dataset",
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width))

val_ds = tf.keras.utils.image_dataset_from_directory(
  "./food_dataset",
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width))

# Abrufen von Klassennamen
class_names = train_ds.class_names


# Sicherstellen, dass die Datasets beim trainieren schnell aufgerufen werden können
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Erstellen der Modelle mit den Neuronalen Netzen
num_classes = len(class_names)

"""
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding="same", activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding="same", activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding="same", activation="relu"),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(num_classes)
])
"""

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding="same", activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding="same", activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding="same", activation="relu"),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(num_classes, name="outputs")
])


# Kompilieren des Modells
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])


# Trainieren des Modells
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# Visualisierung der Trainingsergebnisse
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Trainingsgenauigkeit")
plt.plot(epochs_range, val_acc, label="Validierungsgenauigkeit")
plt.legend(loc="lower right")
plt.title("Trainings- und Validierungsgenauigkeit")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Trainingsverlust")
plt.plot(epochs_range, val_loss, label="Validierungsverlust")
plt.legend(loc="upper right")
plt.title("Trainings- und Validierungsverlust")
plt.show()


# Konvertieren des Modells in ein TFLite-Modell
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# Speichern des TFLite-Modells
date = str(date.today())

with open(date+".tflite", "wb") as f:
  f.write(tflite_model)
