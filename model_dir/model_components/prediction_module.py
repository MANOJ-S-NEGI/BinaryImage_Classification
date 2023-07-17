import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
from model_dir.model_components.generator_file import DataIngestionTraining
import os

# loading the model
load_save_model = tf.keras.models.load_model("saved_model_dir/07_10_2023_16_29_40/model_with_01.weights.best.hdf5")

"""
def testing_data():
    img_dir = "test_images"
    file_names = os.listdir(img_dir)
    filenames = []
    for i in file_names:
        filenames.append(os.path.join(img_dir, i))
    return filenames
"""
# Create a function to import an image and resize it to be able to be used with our model
def load_prep_predict_image(filename, img_shape=150):
    img = plt.imread(filename)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255
    image_expanded = tf.expand_dims(img, axis=0)
    pred = load_save_model.predict(image_expanded)
    class_label = DataIngestionTraining().Label_Classification()  # Get the predicted class
    predict_class = class_label[int(tf.round(pred)[0][0])]
    return predict_class

