import datasets.my_wisenet_dataset  # Register `my_dataset`
import tensorflow_datasets as tfds

ds = tfds.load('my_wisenet_dataset')  # `my_dataset` registered