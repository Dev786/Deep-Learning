import tensorflow as tf
import numpy as np
import pickle


class CNN:
    def __init__(self, epochs, iterations, features, labels, image_size, filter_size, num_channels, conv_features=[32, 64], n_maxpool=2, hidden_layer_shape=[100,24,12,6,2]):
        self.epochs = epochs
        self.iterations = iterations
        self.features = features
        self.labels = labels
        self.image_size = image_size
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.conv_features = conv_features
        self.n_maxpool = n_maxpool
        self.hidden_layer_shape = hidden_layer_shape

        self.X_features = tf.placeholder(dtype=tf.float32, shape=(
            None, image_size[0], image_size[1], num_channels))
        self.Y_labels = tf.placeholder(dtype=tf.float32, shape=(None, 2))

        maxpool_output_divisor = pow(self.n_maxpool, len(conv_features))
        self.hidden_input_layer_size = (
            image_size[0] * image_size[1])//maxpool_output_divisor * conv_features[len(conv_features)-1]

    def set_convolution_params(self):
        self.conv_layers_parms = []
        for i in range(0, len(self.conv_features)):
            param = {
                "weight": tf.Variable(tf.random.truncated_normal((self.filter_size, self.filter_size, self.num_channels, self.conv_features[i]), stddev=0.1)),
                "bias": tf.Variable(tf.zeros(shape=self.conv_features[i]))
            }
            self.conv_layers_parms.append(param)

    def set_fully_connected_params(self):
        self.fully_connected_layers_params = []
        self.fully_connected_layers_params.append(
             {
                "weight": tf.Variable(tf.random.truncated_normal((self.hidden_input_layer_size, self.hidden_layer_shape[0]), stddev=0.05)),
                "bias": tf.Variable(tf.zeros(shape=self.hidden_layer_shape[0]))
            }
        )
        for i in range(1, len(self.hidden_layer_shape)):
            param = {
                "weight": tf.Variable(tf.random.truncated_normal((self.hidden_layer_shape[i-1], self.hidden_layer_shape[i]), stddev=0.05)),
                "bias": tf.Variable(tf.zeros(shape=self.hidden_layer_shape[i]))
            }
            self.fully_connected_layers_params.append(param)

    def computer_cnn_network(self):
        self.conv_layers = []
        for i in range(0, len(self.conv_layers_parms)):
            conv2d = tf.nn.conv2d(
                self.features, self.conv_layers_parms[i]["weight"], [1, 1, 1, 1], 'SAME')
            conv_output = tf.nn.bias_add(
                tf.nn.relu(conv2d), self.conv_layers_parms[i]['bias'])
            conv_out_max_pool = tf.nn.max_pool2d(conv_output, self.n_maxpool, [
                                                 1, 2, 2, 1], padding="SAME")
            self.conv_layers.append(conv_out_max_pool)

    def compute_hidden_layers(self):
        self.hidden_layers = []
        final_conv_shape = self.conv_layers[len(
            self.conv_layers)-1].get_shape().as_list()
        final_conv = tf.reshape(self.conv_layers[len(
            self.conv_layers)-1], shape=(final_conv_shape[0], self.hidden_input_layer_size))
        hidden_layer1_output = tf.add(tf.matmul(
            final_conv, self.fully_connected_layers_params[0]['weight']), self.fully_connected_layers_params[0]['bias'])
        self.hidden_layers.append(hidden_layer1_output)

        for i in range(1, len(self.fully_connected_layers_params)):
            self.hidden_layers.append(
                tf.add(tf.matmul(self.hidden_layers[i-1],
                                 self.fully_connected_layers_params[i]['weight']), self.fully_connected_layers_params[i]['bias'])
            )

    def configure_cnn(self, learning_rate):
        self.output = self.hidden_layers[len(self.hidden_layers)-1]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.Y_labels, logits=self.output))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)
        self.softmax_output = tf.nn.softmax(self.output)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
            self.Y_labels, axis=1), tf.argmax(self.softmax_output, axis=1)), dtype=tf.float32))

    def run_cnn(self, learning_rate):
        self.set_convolution_params()
        self.set_fully_connected_params()
        self.computer_cnn_network()
        self.compute_hidden_layers()
        self.configure_cnn(learning_rate)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for _ in range(self.epochs):
                for _ in range(self.iterations):
                    sess.run(self.optimizer, feed_dict={
                             self.X_features: sess.run(self.features), self.Y_labels: sess.run(self.labels)})
                    print(sess.run(self.accuracy, feed_dict={
                        self.X_features: sess.run(self.features), self.Y_labels: sess.run(self.labels)}))


features = np.array([np.reshape(x, (64, 64, 3))
                     for x in pickle.load(open("../input_data.pkl", 'rb'))])

labels = np.eye(2)[np.array(pickle.load(open('../label_data.pkl', 'rb')))]
cnn = CNN(10, 20, tf.cast(features, tf.float32),
          tf.cast(labels, tf.float32), [64, 64, 3], 3, 3)
cnn.run_cnn(0.005)
