import numpy as np
import pickle

def get_feature_labels():
    features = pickle.load(open('input_data.pkl','rb'))
    labels = pickle.load(open('label_data.pkl','rb'))
    labels = np.eye(2)[labels]
    return features,labels

def shuffle_feature_label(features,labels):
    indices = np.random.choice(features.shape[0],features.shape[0])
    return features[indices],labels[indices]

def get_shuffled_feature_labels():
    features,labels = get_feature_labels()
    features,labels = shuffle_feature_label(np.array(features),np.array(labels))
    return features.reshape(80,64,64,3),labels