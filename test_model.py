import tensorflow as tf
import numpy as np

import random
import string


# Erstellen eines Datensatzes, um die Klassennamen zu bekommen
img_height = 180
img_width = 180

temp_ds = tf.keras.utils.image_dataset_from_directory(
    "./food_dataset",
    image_size=(img_height, img_width))

class_names = temp_ds.class_names

# Vorbereiten des Modells
interpreter = tf.lite.Interpreter(model_path="./2024-02-25.tflite")

classify_lite = interpreter.get_signature_runner('serving_default')


# Klassifizierung eines Bildes vo einer eingegebenen URL
print("Enter the URL of an image to classify: ", end="")
url = input()
filename = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
path = tf.keras.utils.get_file(filename, origin=url)

img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions_lite = classify_lite(sequential_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)
