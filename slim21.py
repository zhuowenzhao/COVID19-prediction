"""
Module of SLIM models of LSTM RNNs. Uses the layer LSTM in Keras2.1 to generate 
inherited layers of reduced gated and cell parameters. 
LSTM1
LSTM2
LSTM3
LSTM4
LSTM5
LSTM6
There are more parameter-reduced models as appeared in arXiv.org: 
1) arXiv:1707.04619
2) arXiv:1707.04623
3) arXiv:1707.04626
Prior references modified the recurrent layer in Keras to achieve parameter-reduction 
4) arXiv:1701.03441
and also for the GRU RNNs and the MGU RNNs: 
5) arXiv:1701.05923
6) arXiv:1701.03452

Your contributions are weclome to improve and/or complete this module development 
to the various SLIM (sub-)models

This version shows the correct # of parameters for each LSTM# model. 
Models LSTM1-6.
10 Apr. 2018.
"""

import warnings

from keras.layers import LSTM
from keras.legacy import interfaces
from keras import backend as K
from keras import regularizers
#from keras import activations
#from keras import initializers
from keras.layers import LSTMCell


class LSTMCell1(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 4,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.bias_i = self.bias[:self.units]
        self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias[self.units * 2: self.units * 3]
        self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = self.bias_i
        x_f = self.bias_f
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = self.bias_o

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                                                  self.recurrent_kernel_i))
        f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                                                  self.recurrent_kernel_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                  self.recurrent_kernel_o))
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]
        
class LSTMCell2(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 1,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        #self.bias_i = self.bias[:self.units]
        #self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias
        #self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = 0
        x_f = 0
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = 0

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                                                  self.recurrent_kernel_i))
        f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                                                  self.recurrent_kernel_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                  self.recurrent_kernel_o))
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]       

class LSTMCell3(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 1),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 4,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        #self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        #self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel
        #self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.bias_i = self.bias[:self.units]
        self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias[self.units * 2: self.units * 3]
        self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = self.bias_i
        x_f = self.bias_f
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = self.bias_o

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            #h_tm1_i = h_tm1 * rec_dp_mask[0]
            #h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            #h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            #h_tm1_i = h_tm1
            #h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            #h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i)
        f = self.recurrent_activation(x_f )
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o)
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]
        
class LSTMCell4(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units+3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 1,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel[:, 2: 2+self.units]
        self.recurrent_kernel_o = self.recurrent_kernel[:, -1]

        #self.bias_i = self.bias[:self.units]
        #self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias
        #self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = 0
        x_f = 0
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = 0

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + h_tm1_i * self.recurrent_kernel_i)
        f = self.recurrent_activation(x_f + h_tm1_f * self.recurrent_kernel_f)
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + h_tm1_o * self.recurrent_kernel_o)
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]

class LSTMCell4a(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units+1),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 1,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        #self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel[:, 1:]
        #self.recurrent_kernel_o = self.recurrent_kernel[:, -1]

        #self.bias_i = self.bias[:self.units]
        #self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias
        #self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = 0
        #x_f = 0
        x_c = K.dot(inputs_c, self.kernel_c)
        #x_o = 0

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            #h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            #h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            #h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            #h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + h_tm1_i * self.recurrent_kernel_i)
        f = 0.96
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = 1.0
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]
        
class LSTMCell5(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units+3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 4,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel[:, 2: 2+self.units]
        self.recurrent_kernel_o = self.recurrent_kernel[:, -1]

        self.bias_i = self.bias[:self.units]
        self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias[self.units * 2: self.units * 3]
        self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = self.bias_i
        x_f = self.bias_f
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = self.bias_o

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + h_tm1_i * self.recurrent_kernel_i)
        f = self.recurrent_activation(x_f + h_tm1_f * self.recurrent_kernel_f)
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + h_tm1_o * self.recurrent_kernel_o)
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]
    
class LSTMCell5a(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units+1),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 2,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        #self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel[:, 1:]
        #self.recurrent_kernel_o = self.recurrent_kernel[:, -1]

        self.bias_i = self.bias[:self.units]
        #self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias[self.units:]
        #self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = self.bias_i
        #x_f = self.bias_f
        x_c = K.dot(inputs_c, self.kernel_c)
        #x_o = self.bias_o

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            #h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            #h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            #h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            #h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + h_tm1_i * self.recurrent_kernel_i)
        f = 0.96
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = 1.0
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]
        
class LSTMCell6(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units*1),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 1,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        #self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        #self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel
        #self.recurrent_kernel_o = self.recurrent_kernel[:, -1]

        #self.bias_i = self.bias[:self.units]
        #self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias
        #self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        #x_i = self.bias_i
        #x_f = self.bias_f
        x_c = K.dot(inputs_c, self.kernel_c)
        #x_o = self.bias_o

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            #h_tm1_i = h_tm1 * rec_dp_mask[0]
            #h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            #h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            #h_tm1_i = h_tm1
            #h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            #h_tm1_o = h_tm1
        i = 1.0
        f = 0.4
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                        self.recurrent_kernel_c))
        o = 1.0
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]

class LSTMCell10(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 1,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel[:, 2]
        self.recurrent_kernel_o = self.recurrent_kernel[:, 3]

        #self.bias_i = self.bias[:self.units]
        #self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias
        #self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = 0
        x_f = 0
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = 0

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + h_tm1_i * self.recurrent_kernel_i)
        f = self.recurrent_activation(x_f + h_tm1_f * self.recurrent_kernel_f)
        c = f * c_tm1 + i * self.activation(x_c + h_tm1_c * self.recurrent_kernel_c)
        o = self.recurrent_activation(x_o + h_tm1_o * self.recurrent_kernel_o)
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]

class LSTMCell11(LSTMCell):
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)


        bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 4,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #self.kernel_i = self.kernel[:, :self.units]
        #self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel
        #self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, 0]
        self.recurrent_kernel_f = self.recurrent_kernel[:, 1]
        self.recurrent_kernel_c = self.recurrent_kernel[:, 2]
        self.recurrent_kernel_o = self.recurrent_kernel[:, 3]

        self.bias_i = self.bias[:self.units]
        self.bias_f = self.bias[self.units: self.units * 2]
        self.bias_c = self.bias[self.units * 2: self.units * 3]
        self.bias_o = self.bias[self.units * 3:]

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_c = inputs * dp_mask[2]

        else:
            inputs_c = inputs
            
        x_i = self.bias_i
        x_f = self.bias_f
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = self.bias_o

        x_c = K.bias_add(x_c, self.bias_c)


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + h_tm1_i * self.recurrent_kernel_i)
        f = self.recurrent_activation(x_f + h_tm1_f * self.recurrent_kernel_f)
        c = f * c_tm1 + i * self.activation(x_c + h_tm1_c * self.recurrent_kernel_c)
        o = self.recurrent_activation(x_o + h_tm1_o * self.recurrent_kernel_o)
                                                  
        
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
                
        return h, [h, c]
    
class LSTMs(LSTM):

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 model='LSTM1',
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        if model == 'LSTM1':
            cell = LSTMCell1(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM2':
            cell = LSTMCell2(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model =='LSTM3':
            cell = LSTMCell3(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM4':
            cell = LSTMCell4(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM4a':
            cell = LSTMCell4a(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM5':
            cell = LSTMCell5(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM5a':
            cell = LSTMCell5a(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM6':
            cell = LSTMCell6(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM10':
            cell = LSTMCell10(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM11':
            cell = LSTMCell11(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
    

        super(LSTM, self).__init__(cell,             # super of lstm not lstms
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)


                    
def _generate_dropout_ones(inputs, dims):
    # Currently, CNTK can't instantiate `ones` with symbolic shapes.
    # Will update workaround once CNTK supports it.
    if K.backend() == 'cntk':
        ones = K.ones_like(K.reshape(inputs[:, 0], (-1, 1)))
        return K.tile(ones, (1, dims))
    else:
        return K.ones((K.shape(inputs)[0], dims))


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)    
