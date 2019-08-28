import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import cv2

output_size = 2
learning_rate = 0.05
epochs = 5
test_size = 0.05
iterations = 10


features = np.array(pickle.load(open('input_data.pkl', 'rb')))
labels = np.array(pickle.load(open('label_data.pkl', 'rb')))

input_size = features.shape[1]
# print(input_size)
one_hot_labels = np.eye(output_size)[labels]


X_data = tf.placeholder(dtype=tf.float32, shape=(None, input_size))
Y_data = tf.placeholder(dtype=tf.float32, shape=(None, 2))

# creating variables for tf models
# print(input_size)
hidden_layers = [
    {
        "weight": tf.Variable(tf.random.normal(shape=(input_size, 3072))),
        "bias": tf.Variable(tf.zeros(shape=(3072)))
    },
    {
        "weight": tf.Variable(tf.random.normal(shape=(3072, 768))),
        "bias": tf.Variable(tf.zeros(shape=(768)))
    },
    {
        "weight": tf.Variable(tf.random.normal(shape=(768, 192))),
        "bias": tf.Variable(tf.zeros(shape=(192)))
    },
    {
        "weight": tf.Variable(tf.random.normal(shape=(192, 48))),
        "bias": tf.Variable(tf.zeros(shape=(48)))
    },
    {
        "weight": tf.Variable(tf.random.normal(shape=(48, 2))),
        "bias": tf.Variable(tf.zeros(shape=(2)))
    }
]

hidden_out_1 = tf.nn.leaky_relu(
    tf.add(tf.matmul(X_data, hidden_layers[0]
                     ['weight']), hidden_layers[0]['bias']))

hidden_out_2 = tf.nn.leaky_relu(
    tf.add(tf.matmul(hidden_out_1,
                     hidden_layers[1]['weight']), hidden_layers[1]['bias']))


hidden_out_3 = tf.nn.leaky_relu(
    tf.add(tf.matmul(hidden_out_2,
                     hidden_layers[2]['weight']), hidden_layers[2]['bias']))


hidden_out_4 = tf.nn.leaky_relu(
    tf.add(tf.matmul(hidden_out_3,
                     hidden_layers[3]['weight']), hidden_layers[3]['bias']))


output = tf.add(tf.matmul(hidden_out_4,
                          hidden_layers[4]['weight']), hidden_layers[4]['bias'])

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y_data)

cost = tf.reduce_mean(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1),
                                           tf.argmax(Y_data, axis=1)), dtype=tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

predictor = tf.nn.sigmoid(output)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    test_X = features[[10,20,25,41,48,56]]
    test_Y = one_hot_labels[[10,20,25,41,48,56]]
    for _ in range(0, epochs):
        cost_value = 0
        accuracy_train = 0
        accuracy_test = 0
        X_Train, X_test, Y_Train, Y_test = train_test_split(
            features, one_hot_labels, test_size=test_size)

        for _ in range(0, iterations):
            sess.run(optimizer, feed_dict={X_data: X_Train, Y_data: Y_Train})
        cost_value = sess.run(
            cost, feed_dict={X_data: X_Train, Y_data: Y_Train})

        accuracy_train = sess.run(
            accuracy, feed_dict={X_data: X_Train, Y_data: Y_Train})

        accuracy_test = sess.run(
            accuracy, feed_dict={X_data: X_test, Y_data: Y_test})

        print("Cost: {}, Train Accuracy: {}, Test Accuracy: {}".format(
            cost_value, accuracy_train, accuracy_test))
    
    # Validating the prediction
    print("True: {}, Predicted: {}".format(np.argmax(test_Y,axis=1),
                                           np.argmax(sess.run(predictor, feed_dict={X_data: test_X}),axis=1)))
