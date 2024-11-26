import os
import pickle


# loading pkl data
def load_data(data_dir, filename):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)


# saving pkl data
def save_data(data_dir, filename, data):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
