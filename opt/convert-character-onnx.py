import tensorflow_datasets as tfds
import tensorflow as tf
import tf2onnx
import onnx

emnist_train = tfds.load(name="emnist/balanced", split="train")

model_path = "models/models.onnx"
print(emnist_train)
# convert to onnx models
onnx_model, _ = tf2onnx.convert.from_keras(emnist_train,
                                           input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32)])
tf.saved_model.save(emnist_train, model_path)
