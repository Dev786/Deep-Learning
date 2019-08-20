import numpy as np
import os
import pickle
import cv2


def preprocess_x(data):
    min_d = np.min(data)
    max_d = np.max(data)
    data = (data - min_d)/(max_d - min_d)
    return data

def create_data_set():
    folders = os.listdir()
    X = []
    Y = []

    for folder in folders:
        if os.path.isdir(os.path.curdir + "/" + folder):
            label = 1 if folder == 'Glaucoma' else 0
            print(label)
            contents = os.listdir(folder)
            os.chdir(os.path.curdir + "/" + folder)
            for content in contents:
                img = cv2.imread(content)
                gray_scale = np.array(cv2.resize(img,dsize=(64,64),interpolation=cv2.INTER_LINEAR))
                gray_scale = gray_scale.flatten()
                X.append(gray_scale)
                Y.append(label)
            os.chdir('../')
    # print(Y[0])
    # X = [preprocess_x(d) for d in X]
    print(X[0])
    input_file = open('input_data.pkl', 'wb')
    label_file = open('label_data.pkl', 'wb')
    pickle.dump(X, input_file)
    pickle.dump(Y, label_file)


create_data_set()
