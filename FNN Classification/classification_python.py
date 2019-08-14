import pandas as pd
import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class FNN:
    def __init__(self, parent_folder,learning_rate,epochs,iterations,batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.iteration = iterations
        self.batch_size = batch_size
    
        os.chdir(parent_folder)
        self.X_data = tf.placeholder(dtype=tf.float32, shape=(None, 784))
        self.Y_data = tf.placeholder(dtype=tf.float32, shape=(None, 10))
        self.hidden_layers = [
            {
                "weight": tf.Variable(tf.random.normal(shape=(784, 1568))),
                "bias": tf.zeros(shape=(1568))
            },
            {
                "weight": tf.Variable(tf.random.normal(shape=(1568, 784))),
                "bias": tf.zeros(shape=(784))
            },
            {
                "weight": tf.Variable(tf.random.normal(shape=(784, 392))),
                "bias": tf.zeros(shape=(392))
            },
            {
                "weight": tf.Variable(tf.random.normal(shape=(392, 196))),
                "bias": tf.zeros(shape=(196))
            },
            {
                "weight": tf.Variable(tf.random.normal(shape=(196, 49))),
                "bias": tf.zeros(shape=(49))
            },
            {
                "weight": tf.Variable(tf.random.normal(shape=(49, 10))),
                "bias": tf.zeros(shape=(10))
            }
        ]

        print(self.hidden_layers[0])
        self.hidden_layer_1_output = tf.add(
            tf.matmul(self.X_data, self.hidden_layers[0]["weight"]), self.hidden_layers[0]["bias"])
        self.hidden_layer_1_output = tf.nn.tanh(self.hidden_layer_1_output)

        self.hidden_layer_2_output = tf.add(tf.matmul(
            self.hidden_layer_1_output, self.hidden_layers[1]["weight"]), self.hidden_layers[1]["bias"])
        self.hidden_layer_2_output = tf.nn.tanh(self.hidden_layer_2_output)

        self.hidden_layer_3_output = tf.add(tf.matmul(
            self.hidden_layer_2_output, self.hidden_layers[2]["weight"]), self.hidden_layers[2]["bias"])
        self.hidden_layer_3_output = tf.nn.tanh(self.hidden_layer_3_output)

        self.hidden_layer_4_output = tf.add(tf.matmul(
            self.hidden_layer_3_output, self.hidden_layers[3]["weight"]), self.hidden_layers[3]["bias"])
        self.hidden_layer_4_output = tf.nn.tanh(self.hidden_layer_4_output)

        self.hidden_layer_5_output = tf.add(tf.matmul(
            self.hidden_layer_4_output, self.hidden_layers[4]["weight"]), self.hidden_layers[4]["bias"])
        self.hidden_layer_5_output = tf.nn.tanh(self.hidden_layer_5_output)

        self.predicted_value = tf.nn.softmax(tf.add(tf.matmul(
            self.hidden_layer_5_output, self.hidden_layers[5]["weight"]), self.hidden_layers[5]["bias"]))

        self.cost = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.predicted_value, labels=self.Y_data)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.equal = tf.equal(
            tf.argmax(self.predicted_value, axis=1), tf.argmax(self.Y_data, axis=1))

        self.accuracy = tf.reduce_mean(tf.cast(self.equal, tf.float32))

        self.init = tf.global_variables_initializer()

    def get_all_images(self):
        all_dir = os.listdir()
        input_data = []
        label_data = []
        for directory in all_dir:
            os.chdir(directory)
            image_dir = os.listdir()
            for image_name in image_dir:
                img = cv2.imread(image_name, cv2.COLOR_RGB2GRAY)
                # scale_percentage = 80
                width = 28  # int(img.shape[1] * scale_percentage / 100)
                height = 28  # int(img.shape[0]*scale_percentage / 100)
                dim = (width, height)
                resized = np.array(cv2.cvtColor(cv2.resize(
                    img, dim, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
                input_data.append(resized.flatten())
                label_data.append(directory)
                # print(len(input_data))
            os.chdir("../")

        one_hot_encoder = LabelEncoder()
        labels = np.eye(10)[one_hot_encoder.fit_transform(np.array(label_data).reshape(-1,1))]
        return input_data, labels

    def train_neural_network(self):
        X,Y = fnn.get_all_images()
        X = np.array(X)
        Y = np.array(Y)
        sess = tf.Session()
        sess.run(self.init)

        for i in range(0,self.epochs):
            indices = np.random.choice(len(X),self.batch_size).astype(int)
            
            for _ in range(0,self.iteration):
                sess.run(self.optimizer,feed_dict={self.X_data:X[indices][:],self.Y_data:Y[indices][:]})
            
                # cost = sess.run(self.cost,feed_dict={self.X_data:X_train,self.Y_data:Y_train})
            accuracy = sess.run(self.accuracy,feed_dict={self.X_data:X[indices][:],self.Y_data:Y[indices][:]})
            if i%50 == 0:
                print("Training Accuracy: ",accuracy)

fnn = FNN('./images',0.05,1000,10,20)
fnn.train_neural_network()

