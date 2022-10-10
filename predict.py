import tensorflow as tf
import cv2
import numpy as np

# oh my god
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
# print(input_shape)
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
#r"C:\Users\fdimo\Desktop\coral_images\
input_data = cv2.imread(r"predict-[FINAL].png", cv2.IMREAD_UNCHANGED)
input_data = input_data.astype(np.float32)
# print(interpreter.get_input_details())
input_data = np.expand_dims(input_data, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)

class_names = ['can', 'paper', 'plastic', 'tissues']
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print(class_names[np.argmax(output_data[0])])
#
# predictions = output_data
# score = tf.nn.softmax(predictions[0])
#
# print(score)
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
#
# print(np.argmax(score), 100 * np.max(score))
# print(output_data)
