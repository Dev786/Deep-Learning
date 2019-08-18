import pickle
import numpy as np

features = np.array(pickle.load(open('input_data.pkl', 'rb')))
labels = np.array(pickle.load(open('label_data.pkl', 'rb'))).reshape(-1,1)

print(labels)