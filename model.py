import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import numpy as np

def matmul_1d(a, b):
    return tf.einsum('ij,j->i', a, b)
def batch_matmul_1d(a, b):
    return tf.einsum('ij,kj->ki', a, b)

class Activity(keras.layers.Layer):
    def __init__(self, units, batch_size, alpha, sparsity_coef):
        super(Activity, self).__init__()
        self.w_init = tf.random_normal_initializer()
        self.batch_size = batch_size
        self.units = units
        self.w = tf.Variable(
            initial_value=self.w_init(shape=(batch_size, units), dtype="float32"),
            trainable=False,
        )
        self.rate = alpha
        self.sparsity_coef = sparsity_coef
    def shrink(self):
        a = self.w
        b = self.rate * self.sparsity_coef
        prior_shrink = tf.abs(a) - b
        shrink_positive = tf.clip_by_value(prior_shrink, 0, np.inf)
        sign_a = tf.math.sign(a)
        a.assign(sign_a * shrink_positive)
    def update(self, dictionary, x):
        tf.debugging.assert_all_finite(self.w, 'Check activity weights finite')
        batch_size = x.shape[0]
        
        self.w[:batch_size].assign(self.w[:batch_size]-self.rate* \
            tf.einsum('ij, ki->kj', dictionary, (tf.einsum('ij,kj->ki', dictionary, self.w[:batch_size])-x)))
        self.shrink()
    def reset(self):
        self.w.assign(self.w_init(shape=(self.batch_size, self.units), dtype="float32"))
    def call(self, dictionary, batch_size):
        return batch_matmul_1d(dictionary, self.w[:batch_size])

class Dictionary(keras.layers.Layer):
    def __init__(self, units, dict_filter_size, beta):
        super(Dictionary, self).__init__()
        w_init = tf.random_normal_initializer()
        self.units = units
        self.beta = beta
        self.A = tf.Variable(
            initial_value=w_init(shape=(units, units), dtype="float32"),
            trainable=False,
        )
        self.B = tf.Variable(
            initial_value=w_init(shape=(dict_filter_size, units), dtype="float32"),
            trainable=False,
        )
        self.w =  tf.Variable(
            initial_value=w_init(shape=(dict_filter_size, units), dtype="float32"),
            trainable=False,
        )
    def call(self, inputs, activity):
        return tf.matmul(self.dictionary, activity)
    def update_AB(self, activity, x):
        tf.debugging.assert_all_finite(self.w, 'Check A finite')
        tf.debugging.assert_all_finite(self.w, 'Check B finite')
        batch_size = x.shape[0]
        self.A.assign(self.beta*self.A + (1-self.beta)*tf.reduce_mean(\
            tf.einsum('ij,ik->ijk', activity[:batch_size], activity[:batch_size]), axis=0))
        self.B.assign(self.beta*self.B + (1-self.beta)*tf.reduce_mean(tf.einsum('ij,ik->ijk', x, activity[:batch_size]), axis=0))
    def update(self):
        tf.debugging.assert_all_finite(self.w, 'Check dictionary weights finite')
        epsilon = 1e-5
        for i in range(self.units):
            self.w[:, i].assign(1/(self.A[i, i]+epsilon) * (self.B[:, i] - tf.einsum('ij,j->i',self.w,self.A[:, i])+self.w[:, i]*self.A[i, i]))
            self.w[:, i].assign(self.w[:, i] / tf.norm(self.w[:, i])+epsilon)

def dictionary_loss(dictionary, activity, x):
    return tf.reduce_mean(0.5*tf.square(tf.einsum('ij,kj->ki', dictionary, activity) - x))
def sparsity_loss(activity, sparsity_coef):
    return tf.abs(activity)*sparsity_coef

class SparseModel(keras.Model):
    def __init__(self, activity, dictionary, activity_epochs, dict_filter_size, \
                data_size, batch_size, num_layers):
        super(SparseModel, self).__init__()
        self.activity = activity
        self.dictionary = dictionary
        self.dict_filter_size = dict_filter_size
        self.data_size = data_size
        self.batch_size = batch_size
        self.activity_epochs = activity_epochs
        self.batch_num = data_size // batch_size + (1 if (self.data_size % self.batch_size) else 0)
        self.num_layers = num_layers
    def compile(self, sparsity_loss, dictionary_loss):
        super(SparseModel, self).compile()
        self.sparsity_loss = sparsity_loss
        self.dictionary_loss = dictionary_loss
    @tf.function
    def train_step_end(self):
        if self._train_counter % self.batch_num == 0:
            self.dictionary.update()
    def train_step(self, patches):
        patches = tf.cast(patches, dtype=tf.float32)
        batch_size = patches.shape[0]
        patches = tf.reshape(patches, [-1, self.dict_filter_size])
        self.activity.reset()
        
        dictionary = self.dictionary.w
        activity = self.activity.w
        for _ in range(self.activity_epochs):
            self.activity.update(dictionary, patches)
        dictionary_loss = self.dictionary_loss(dictionary, activity, patches)
        tf.debugging.assert_all_finite(dictionary_loss, 'Check dictionary loss')
        sparsity_loss = self.sparsity_loss(activity, self.activity.rate*self.activity.sparsity_coef)
        tf.debugging.assert_all_finite(sparsity_loss, 'Check sparsity loss')
        self.dictionary.update_AB(activity, patches)
        self.train_step_end()
        return {'dictionary loss': dictionary_loss, 'sparsity loss': sparsity_loss}
    def call(self, patches):
        patches = tf.cast(patches, dtype=tf.float32)
        patches = tf.reshape(patches, [-1, self.dict_filter_size])
        self.activity.reset()

        dictionary = self.dictionary.w
        for _ in range(self.activity_epochs):
            self.activity.update(dictionary, patches)
        return self.activity.call(self.dictionary.w, batch_size=patches.shape[0])