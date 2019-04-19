import tensorflow as tf
import numpy as np



class Actor:
    def __init__(self, sess,
                 input_size=None,
                 output_size=None,
                 action_bounds=None,
                 batch_size=20,
                 lr=0.001,
                 TAU=0.001):
        self.sess = sess
        self.TAU = TAU
        self.action_bound = action_bounds
        self.inputs, self.actions, self.scaled_actions = self.__build_model(input_size, output_size, name='Actor')
        self.target_inputs, self.target_actions, self.target_scaled_actions = self.__build_model(input_size, output_size, name='Actor_target')
        
        self.weights = tf.trainable_variables(scope='Actor')
        self.target_weights = tf.trainable_variables(scope='Actor_target')
        

        self.q_grads_wrt_actions = tf.placeholder(tf.float32,
                                                  shape=[None, output_size])
        grads = tf.gradients(self.scaled_actions, self.weights, -self.q_grads_wrt_actions)

        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(grads, self.weights))

        self.train_target = \
                          [self.target_weights[i].assign(tf.multiply(self.weights[i], self.TAU) + tf.multiply(self.target_weights[i], 1-self.TAU)) for i in range(len(self.target_weights))]

        self.sess.run(tf.global_variables_initializer())

    def __build_model(self, input_size, output_size, name=None):
        with tf.variable_scope(name):
            inputs = tf.placeholder(tf.float32, shape=(None, input_size))
            layer1 = dense_layer(inputs,
                                 input_size, 20,
                                 activation=tf.nn.relu,
                                 initializer=tf.initializers.random_normal,
                                 name='dense_1')
            layer2 = dense_layer(layer1,
                                 20, 20,
                                 activation=tf.nn.relu,
                                 initializer=tf.initializers.random_normal,
                                 name='dense_2')
            actions = dense_layer(layer2,
                                  20, output_size,
                                  activation=tf.nn.tanh,
                                  initializer=tf.initializers.random_normal,
                                  name='actions')
            scaled_actions = tf.clip_by_value(actions,
                                              -self.action_bound,
                                              self.action_bound)

            tf.summary.tensor_summary('actions', actions)
            tf.summary.merge_all()
            
            return inputs, actions, scaled_actions

    def train(self, inputs, grads):
        return self.sess.run(self.optimizer, feed_dict={self.inputs: inputs, self.q_grads_wrt_actions: grads})

    def predict(self, inputs):
        return self.sess.run(self.scaled_actions, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_actions, feed_dict={self.target_inputs: inputs})

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
