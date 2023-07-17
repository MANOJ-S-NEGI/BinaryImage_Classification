MAIN_DATA_DIR = "C:/Users/Manoj Negi/PycharmProjects/pythonProject/pythonProject/cat_vs_dog"
TRAIN_DATA_DIR = "C:/Users/Manoj Negi/PycharmProjects/pythonProject/pythonProject/cat_vs_dog/train"
VALIDATION_DATA_DIR = "C:/Users/Manoj Negi/PycharmProjects/pythonProject/pythonProject/cat_vs_dog/validation"

# geneerator module constants:

SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
RESCALE = 1/255
SHUFFLE = True
TARGET_SIZE = 150
CLASS_MODE = 'binary'
BATCH_SIZE = 32

# TRAINING MODULE CONSTANTS:
KERNEL_SIZE = (3, 3)
PADDING = 'same'
INPUT_SHAPE = (150, 150, 3)
POOL_SIZE = (2, 2)
VERBOSE = 2
SAVE_MODEL_PATH = "saved_model_dir"
SAVED_MODEL_NAME = "model.weights.best.hdf5"
CALLBACK_PATIENCE = 10
EPOCHS = 50
