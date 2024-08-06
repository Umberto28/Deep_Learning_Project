"""
Before training and evaluation - make sure to select desired/correct settings
"""

# Input dataset related settings
DATA_PATH = "Easter2-main/IAM/"
DATA_PATH_V2= "Easter2-main/data/NoAugmentation"

INPUT_HEIGHT = 80
INPUT_WIDTH = 2000
INPUT_SHAPE = (INPUT_WIDTH, INPUT_HEIGHT)

TACO_AUGMENTAION_FRACTION = 0.9

# If Long lines augmentation is needed (see paper)

LONG_LINES = False
LONG_LINES_FRACTION = 0.3

# Model training parameters
BATCH_SIZE = 32
EPOCHS = 150
VOCAB_SIZE = 80
DROPOUT = True
OUTPUT_SHAPE = 500

# Initializing weights from pre-trained 
LOAD = False
LOAD_CHECKPOINT_PATH = "Easter2-main\weights\saved_checkpoint.hdf5"

# Other learning parametes
LEARNING_RATE = 0.01
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997

# Checkpoints parametes
CHECKPOINT_PATH = 'Easter2-main\weights/EASTER2--{epoch:02d}--{loss:.02f}.hdf5'
LOGS_DIR = 'Easter2-main\logs'
BEST_MODEL_PATH = "Easter2-main\weights/saved_checkpoint.hdf5"