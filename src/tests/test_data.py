import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

def test_data():
    dataset_path = "datasets"
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    assert len(train_X) == 60000
    assert len(test_X) == 10000

    assert train_X.shape == (60000, 28, 28)
    assert test_X.shape == (10000, 28, 28)
