####### Importing Libraries
import numpy as np
import tensorflow as tf
from MViT_MHSA import MViT_MHSA

####### MViT Encoder
class MViT_Encoder(tf.keras.layers.Layer):

    """
    MViT Encoder
    """

    def __init__(self,
                 d_in,
                 d_out,
                 num_heads,
                 q_kernel,
                 q_stride,
                 kv_kernel,
                 kv_stride,
                 rate,
                 dff_dim):
        
        ##### Defining essentials
        super().__init__()
        self.d_in = d_in # Input dimensions
        self.d_out = d_out # Model Embedding Dimensions: Soft Attention requires d_out // num_heads = 0
        self.num_heads = num_heads # Number of attention heads
        self.q_kernel = q_kernel # Pooling kernel for Query
        self.q_stride = q_stride # Stride for Query
        self.kv_kernel = kv_kernel # Pooling kernel for Keys and Value
        self.kv_stride = kv_stride # Stride for Keys and Value
        self.dff_dim = dff_dim # Feed-forward network dimension 
        self.rate = rate # Rate of the dropout
    
        ##### Defining layers
        self.attn = MViT_MHSA(self.d_in,
                              self.d_out,
                              self.num_heads,
                              self.q_kernel,
                              self.q_stride,
                              self.kv_kernel,
                              self.kv_stride)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff_dim, activation="relu"),
            tf.keras.layers.Dense(self.d_out),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_in': self.d_in,
            'd_out': self.d_out, 
            'num_heads': self.num_heads,
            'q_kernel': self.q_kernel,
            'kv_kernel': self.kv_kernel,
            'q_stride': self.q_stride,
            'kv_stride': self.kv_stride,
            'dff_dim': self.dff_dim,
            'rate': self.rate
        })
        return config

    def call(self,x,thw_shape,training): 

        """
        MViT Encoder

        INPUTS:-
        1) x: Input of shape (B,L,d_in)
        2) thw_shape: Original input dimensions [T,H,W]

        OUTPUTS:-
        1) x: Output of shape (B,L_q,d_out)
        2) q_shape: Output's video shape [T_q,H_q,W_q]
        """

        x, q_shape = self.attn(x,thw_shape) # Attention block
        x = self.dropout1(x, training=training) # Dropout
        x = self.layernorm1(x) # Layer normalization

        x_ffn = self.ffn(x) # Feed-forward network
        x_ffn = self.dropout2(x_ffn, training=training)
        return self.layernorm2(x_ffn + x), q_shape

###### Testing
#a = tf.random.normal(shape=(25,320,32))
#b = tf.random.normal(shape=(25,40,64))
#thw_shape = [10,8,4]
#mvit_block = MViT_Encoder(32,64,4,(2,2,2),(2,2,2),
#                          (1,1,1),(1,1,1),
#                          rate=0.3,dff_dim=128)
#b, b_shape = mvit_block(a,thw_shape)
#print(b.shape, b_shape)

#Input_layer = tf.keras.layers.Input(shape=(320,32))
#output_layer, _ = mvit_block(Input_layer,thw_shape)
#model = tf.keras.models.Model(inputs=Input_layer,outputs=output_layer)
#model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='mse')
#model.summary()
#model.fit(a,b,epochs=100)