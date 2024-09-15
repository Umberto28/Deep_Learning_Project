import os
import tensorflow as tf
from predict import test_on_iam
from easter_model import train

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# train()

checkpoint_path = "Easter2-main\weights\EASTER2--150--7.90.hdf5"
# checkpoint_path = "Empty"

test_on_iam(show=False, partition="validation", checkpoint=checkpoint_path, uncased=True)
# test_on_iam(show=False, partition="test", checkpoint=checkpoint_path, uncased=True)

# checkpoint = tf.keras.models.load_model(
#             checkpoint_path,
#             custom_objects={'<lambda>': lambda x, y: y,
#             'tensorflow':tf, 'K':tf.keras.backend}
#         )