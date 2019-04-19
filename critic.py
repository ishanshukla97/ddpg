import tensorflow as tf
import numpy as np



class Critic:
    def __init__(self, sess,
                 input_size=None,
                 output_size=None,
                 batch_size=20,
                 lr=0.0001,
                 TAU=0.001):
        self.sess = sess
        self.TAU = TAU
        self.input_s, self.input_a, self.preds = self.__build_model(input_size, output_size, name='Critic')
        self.target_input_s, self.target_input_a, self.target_preds = self.__build_model(input_size, output_size, name='Critic_target')
        
        self.weights = tf.trainable_variables(scope='Critic')
        self.target_weights = tf.trainable_variables(scope='Critic_target')
        
        self.labels = tf.placeholder(tf.float32, shape=[None, output_size])

        self.q_grads_wrt_actions = tf.gradients(self.preds, self.input_a)

        self.loss = tf.losses.mean_squared_error(self.labels, self.preds)
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

        self.train_target = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.TAU) + tf.multiply(self.target_weights[i], 1-self.TAU))
                             for i in range(len(self.target_weights))]

        self.sess.run(tf.global_variables_initializer())

    def __build_model(self, input_size, output_size, name=None):
        with tf.variable_scope(name):
            input_s = tf.placeholder(tf.float32, shape=(None, input_size[0]))
            input_a = tf.placeholder(tf.float32, shape=(None, input_size[1]))
            
            layer1 = dense_layer(input_s,
                                 input_size[0], 20,
                                 activation=tf.nn.relu,
                                 initializer=tf.initializers.random_normal,
                                 name='dense_1')
            layer2 = dense_layer(layer1,
                                 20, 20,
                                 activation=tf.nn.relu,
                                 initializer=tf.initializers.random_normal,
                                 name='dense_2')
            layer_action = dense_layer(input_a,
                                       input_size[1], 20,
                                       activation=tf.nn.relu,
                                       initializer=tf.initializers.random_normal,
                                       name='dense_action')
            layer_merge = tf.concat([layer2, layer_action], axis=1)

            q_value = dense_layer(layer_merge,
                                  40, output_size,
                                  activation=tf.nn.relu,
                                  initializer=tf.initializers.random_normal,
                                  name='dense_4')

            tf.summary.scalar('Q_value', q_value)
            tf.summary.merge_all()
            
            return input_s, input_a, q_value

    def gradients(self, states, actions):
        return self.sess.run(self.q_grads_wrt_actions, feed_dict={self.input_s: states, self.input_a: actions})

    def train(self, states, actions, targets):
        return self.sess.run(self.optimizer, feed_dict={self.input_s: states, self.input_a: actions, self.labels: targets})

    def predict(self, states, actions):
        return self.sess.run(self.preds, feed_dict={self.input_s: states, self.input_a: actions})

    def predict_target(self, states, actions):
        return self.sess.run(self.target_preds, feed_dict={self.target_input_s: states, self.target_input_a: actions})

    def train_target(self):
        self.sess.run(self.train_target)

def dense_layer(x, in_units, out_units, activation=None, initializer=None, name=None):
    W_name = name + '/weights'
    b_name = name + '/bias'
    
    W = tf.get_variable(W_name,
                        [in_units, out_units],
                        dtype=tf.float32,
                        initializer=initializer)
    b = tf.get_variable(b_name,
                        [out_units],
                        dtype=tf.float32,
                        initializer=initializer)

    out = tf.add(tf.matmul(x, W), b)

    if activation is not None:
        out = activation(out)

    tf.summary.histogram(W_name, W)
    tf.summary.histogram(b_name, b)

    return out
