import tensorflow as tf
from predict import test_on_iam
from easter_model import train

# train()


checkpoint_path = "Easter2-main\weights\EASTER2--150--7.90.hdf5"

test_on_iam(show=False, partition="validation", checkpoint=checkpoint_path, uncased=True)
# test_on_iam(show=False, partition="test", checkpoint=checkpoint_path, uncased=True)

# checkpoint = tf.keras.models.load_model(
#             checkpoint_path,
#             custom_objects={'<lambda>': lambda x, y: y,
#             'tensorflow':tf, 'K':tf.keras.backend}
#         )