import os
import numpy as np

def save(filepath, data):
    """
    Function to save the preprocessed data into a folder structure
    :param filepath: string - path of the file to save
    :param data: list of preprocessed data
    :return: Nothing
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, np.array(data, dtype=object), allow_pickle=True)