import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from keras.api._v2.keras import Sequential
from keras.api._v2.keras import layers

import tempfile

print(tf.__version__)

batch_size = 5
img_height = 480
img_width = 640

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     r"C:\Users\fdimo\Desktop\coral_images\joint",
#     validation_split=0.2,
#     subset="training",
#     color_mode="rgba",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     r"C:\Users\fdimo\Desktop\coral_images\joint",
#     validation_split=0.2,
#     subset="validation",
#     color_mode="rgba",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)
#
# class_names = train_ds.class_names
# print(class_names)
#
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break
#
# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# normalization_layer = layers.Rescaling(1. / 255)
#
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
#
# num_classes = len(class_names)
#
# # data_augmentation = keras.Sequential(
# #  [
# #    layers.RandomFlip("horizontal",
# #                      input_shape=(img_height,
# #                                  img_width,
# #                                  3)),
# #    layers.RandomRotation(0.1),
# #    layers.RandomZoom(0.1),
# #  ]
# # )
#
# # plt.figure(figsize=(10, 10))
# # for images, _ in train_ds.take(1):
# #    for i in range(9):
# #        augmented_images = data_augmentation(images)
# #        ax = plt.subplot(3, 3, i + 1)
# #        plt.imshow(augmented_images[0].numpy().astype("uint8"))
# #        plt.axis("off")
#
#
# model = Sequential([
#     #layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 4)),
#     layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 4)),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
# ])
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.summary()
#
# epochs = 20
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs
# )
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
#
# # #MODEL_DIR = tempfile.gettempdir()
MODEL_DIR = r"tf_model"
version = 1
export_path = os.path.join(MODEL_DIR, str(version))


# # print('export_path = {}\n'.format(export_path))
# #
# # tf.keras.models.save_model(
# #     model,
# #     export_path,
# #     overwrite=True,
# #     include_optimizer=True,
# #     save_format=None,
# #     signatures=None,
# #     options=None
# # )
# #
#
# model.save(export_path)
# print("Model saved")
#
#
# Convert the model

#oh my god

def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(r"C:\Users\fdimo\Desktop\coral_images\joint\*\*")
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=4)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]


converter = tf.lite.TFLiteConverter.from_saved_model(export_path)  # path to the SavedModel directory
# # This enables quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # This sets the representative dataset for quantization
# converter.representative_dataset = representative_data_gen
# # This ensures that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
# converter.target_spec.supported_types = [tf.int8]
# # These set the input and output tensors to uint8 (added in r2.3)
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted")

new_path = r"3-[FINAL].png"

img = tf.keras.utils.load_img(
    new_path, target_size=(img_height, img_width), color_mode="rgba"
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

model = tf.keras.models.load_model(
    export_path, custom_objects=None, compile=True, options=None
)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ['background', 'metal', 'paper', 'plastic']

print(class_names)
print(predictions)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

print(np.argmax(score), 100 * np.max(score))
