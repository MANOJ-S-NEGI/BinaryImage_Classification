import tensorflow as tf
from model_dir.model_components.generator_file import DataIngestionTraining
from model_dir.log_dir import logging
from model_dir import constants
import os
from datetime import datetime


class TrainingConfig:
    def __init__(self):
        self.train_data = DataIngestionTraining().Train_DataGenerator()
        self.validation_data = DataIngestionTraining().Validation_DataGenerator()

    @staticmethod
    def Model_Layer():
        try:
            logging.info("Initialising the model layering")
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, kernel_size=constants.KERNEL_SIZE, padding=constants.PADDING,
                                       activation="relu", input_shape=constants.INPUT_SHAPE),
                tf.keras.layers.MaxPooling2D(pool_size=constants.POOL_SIZE),
                tf.keras.layers.Conv2D(64, kernel_size=constants.KERNEL_SIZE, padding=constants.PADDING,
                                       activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=constants.POOL_SIZE),
                tf.keras.layers.Conv2D(128, kernel_size=constants.KERNEL_SIZE, padding=constants.PADDING,
                                       activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=constants.POOL_SIZE),
                tf.keras.layers.Conv2D(64, kernel_size=constants.KERNEL_SIZE, padding=constants.PADDING,
                                       activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=constants.POOL_SIZE),
                tf.keras.layers.Conv2D(64, kernel_size=constants.KERNEL_SIZE, padding=constants.PADDING,
                                       activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=constants.POOL_SIZE),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
            model.summary()
            return model
        except Exception as e:
            raise Exception("error in layer function ", str(e))

    @staticmethod
    def callbacks():
        try:
            logging.info("initialising the callback functions")
            # Setting the Callback Function -
            # Create a function to implement a ModelCheckpoint callback with a specific filename

            timestamp = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
            saved_model_directory = os.path.join(constants.SAVE_MODEL_PATH, timestamp)
            os.makedirs(saved_model_directory, exist_ok=True)
            filepath = f"{saved_model_directory}/{constants.SAVED_MODEL_NAME}"
            callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                           verbose=constants.VERBOSE,
                                                           save_best_only=True)

            # Create a function to implement an Early stop callback with loss monitor
            Early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=constants.CALLBACK_PATIENCE,
                                                          verbose=constants.VERBOSE)
            return callbacks, Early_stop

        except Exception as e:
            raise Exception("error in callback function ", str(e))

    def model_compile_training(self):
        try:
            model_layers = TrainingConfig.Model_Layer()
            callbacks, early_stop = TrainingConfig.callbacks()

            logging.info("compiling model")
            model_layers.compile(optimizer=tf.keras.Experimental.optimizers.SGD(learning_rate=0.01, momentum=0.9),  loss='binary_crossentropy', metrics=['accuracy'])
            logging.info("model compilation done")
            logging.info("initializing the model fitting/training")

            history_4 = model_layers.fit(self.train_data,
                                         validation_data=self.validation_data,
                                         validation_steps=int(len(self.validation_data)),
                                         steps_per_epoch=int(len(self.train_data)),
                                         batch_size=constants.BATCH_SIZE,
                                         epochs=constants.EPOCHS,
                                         callbacks=[early_stop, callbacks])
            logging.info("model with trainable bias saved")

        except Exception as e:
            raise ("error in model_compile_training function", e)
