import tensorflow as tf


class YoloArchitecture:

    def __init__(self, no_nodes, no_classes):
        self.n_nodes_hl = no_nodes
        self.n_classes = no_classes

    def neural_network_model(self, data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, self.n_nodes_hl[0]])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl[0]]))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl[0], self.n_nodes_hl[1]])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl[1]]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl[1], self.n_nodes_hl[2]])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl[2]]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl[2], self.n_classes])),
                        'biases': tf.Variable(tf.random_normal([self.n_classes])), }

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output
