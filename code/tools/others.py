import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data