from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm


def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value

import pickle

# Load the object from the pickle file
with open('total_loss.pkl', 'rb') as file:
    total_loss = pickle.load(file)

from tensorflow.keras.models import load_model

# Load the trained model
save_model = load_model('trained_model.h5',custom_objects={'dice_loss_plus_1focal_loss': total_loss,'jaccard_coef':jaccard_coef})

image_path = '01d.jpg'

# Open and display the image
img = Image.open(image_path)
img_array = np.array(img)
print('8989',img_array.shape)
cropped_img_array = np.array(img)

# Display the shape of the array (height, width, channels for a color image)
print("Image Shape/*/:", cropped_img_array.shape)
crop_width = crop_height = 256

# Calculate the starting point for the crop
start_x = (img_array.shape[1] - crop_width) // 3
start_y = (img_array.shape[0] - crop_height) // 2

# Crop the image
cropped_img_array = img_array[start_y:start_y + crop_height, start_x:start_x + crop_width, :]
print('*-*-',cropped_img_array.shape)
# Open and display the image

# Assuming image01 is your image data with shape (height, width, channels)
# Reshape the image to (height * width, channels) for scaling
original_shape = cropped_img_array.shape
image01_reshaped = cropped_img_array.reshape(-1, original_shape[-1])

# Initialize the MinMaxScaler
minmaxscaler = MinMaxScaler()

# Fit and transform the reshaped image data
m1 = minmaxscaler.fit_transform(image01_reshaped)


s1 = m1.reshape(original_shape)

test_image_input = np.expand_dims(s1, 0)

prediction = save_model.predict(test_image_input)
predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0,:,:]

plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(s1)
plt.subplot(232)
plt.title("Predicted Image")
plt.imshow(predicted_image)


