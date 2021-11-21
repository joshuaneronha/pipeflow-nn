import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from skimage import transform

def import_data():
    """
    Loads in data and transposes it into proper shapes
    """
    with open('data/dataX.pkl', 'rb') as f:
        load_geoms = pickle.load(f)[:,1,:,:]
        geometries = np.transpose(load_geoms, [0,2,1])
        geometries = np.round(transform.resize(geometries,(geometries.shape[0],80,geometries.shape[2])))
        geometries = np.ones(geometries[:,:,12:].shape) - np.equal(geometries[:,:,12:],0)

    with open('data/dataY.pkl', 'rb') as f:
        outputs = np.transpose(pickle.load(f), [0,1,3,2])
        outputs = transform.resize(outputs,(outputs.shape[0],outputs.shape[1],80,outputs.shape[3]))
        outputs = outputs[:,:,:,12:]

    return np.expand_dims(geometries,3), outputs




# outputs.shape


def get_next_batch(input_array, label_array, start_index, batch_size):
    """
    Accepts an array of inputs and labels along with a starting index and batch
    size in order to separate the full array of data into batches.
    :inputs: a NumPy array of all images with shape (n x 2)
    :labels: a NumPy array of all labels with shape (n x 1)
    :start_index: the first index desired in the batch
    :batch_size: how many total images desired in the batch
    """
    return input_array[start_index: (start_index + batch_size), :], label_array[start_index: (start_index + batch_size)]
