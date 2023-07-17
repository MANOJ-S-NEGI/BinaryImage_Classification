import os
from model_dir import constants
import tensorflow
from model_dir.log_dir import logging


class DataIngestionTraining:
    def __init__(self):
        self.train_path = constants.TRAIN_DATA_DIR
        self.validation_path = constants.VALIDATION_DATA_DIR

    def Label_Classification(self):
        try:
            labels = os.listdir(self.train_path)
            return labels
        except Exception as e:
            raise Exception("error in Label_Classification ", str(e))

    def Train_DataGenerator(self):
        try:
            logging.info("creating Train data generator -get all the pixel values between 1 and 0")
            train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=constants.RESCALE,
                                                                                    shear_range=constants.SHEAR_RANGE,
                                                                                    zoom_range=constants.ZOOM_RANGE,
                                                                                    horizontal_flip=constants.HORIZONTAL_FLIP)

            logging.info("created Train data generator -get all the pixel values between 1 and 0")
            logging.info("importing the data from dir and convert into the batching")
            train_data = train_datagen.flow_from_directory(self.train_path,
                                                           shuffle=constants.SHUFFLE,
                                                           target_size=(constants.TARGET_SIZE, constants.TARGET_SIZE),
                                                           class_mode=constants.CLASS_MODE,
                                                           batch_size=constants.BATCH_SIZE)

            logging.info("Data into batches created as train_data and valid_data")
            return train_data

        except Exception as e:
            raise Exception("error in Train_Validation_DataGenerator ", str(e))

    def Validation_DataGenerator(self):
        try:
            logging.info("creating valid data generator -get all the pixel values between 1 and 0")
            validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=constants.RESCALE,
                                                                                         shear_range=constants.SHEAR_RANGE,
                                                                                         zoom_range=constants.ZOOM_RANGE,
                                                                                         horizontal_flip=constants.HORIZONTAL_FLIP)
            logging.info("created valid data generator -get all the pixel values between 1 and 0")

            logging.info("importing the data from dir and convert into the batching")

            valid_data = validation_datagen.flow_from_directory(self.validation_path,
                                                                shuffle=constants.SHUFFLE,
                                                                target_size=(constants.TARGET_SIZE, constants.TARGET_SIZE),
                                                                class_mode=constants.CLASS_MODE,
                                                                batch_size=constants.BATCH_SIZE, )
            logging.info("Data into batches created valid_data")

            return valid_data

        except Exception as e:
            raise Exception("error in Train_Validation_DataGenerator ", str(e))

# print(DataIngestionTraining().Train_Validation_DataGenerator())
