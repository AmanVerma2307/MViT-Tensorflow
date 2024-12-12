###### Importing Libraries
import tensorflow as tf

###### Video Vision Transformer

##### Tubelet Embedding
class Tubelet_Embedding(tf.keras.layers.Layer):

    def __init__(self, embed_dim, patch_size):

        #### Defining Essentials
        super().__init__()
        self.embed_dim = embed_dim # Embedding Dimensions 
        self.patch_size = patch_size # A tuple of dimensions - (p_t,p_h,p_w), with each corresponding to patch dimensions

        #### Defining Layers
        self.embedding_layer =  tf.keras.layers.Conv3D(filters=self.embed_dim,
                                                        kernel_size=self.patch_size,
                                                        strides=self.patch_size,
                                                        padding="VALID") # Tubelet Patch and Embedding Creation Layer
        self.flatten =  tf.keras.layers.Reshape((-1,self.embed_dim)) # Layer to Flatten the Patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size
        })

    def call(self,X_in):

        """
        Layer to Project the input spatio-temporal sequence into Tubelet Tokens

        INPUTS:-
        1) X_in: Input video sequence of dimensions (T,H,W,C)

        OUTPUTS:-
        1) X_o: Tubelet Embeddings of shape (n_t*n_h*n_w,embed_dim)
        
        """
        #### Tubelet Embedding Creation
        X_o = self.embedding_layer(X_in) # Embedding Layer
        X_o = self.flatten(X_o) # Flattening Input

        return X_o

###### Positional Embedding
class PositionEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, maxlen, embed_dim):

        #### Defining Essentials
        super().__init__()
        self.maxlen = maxlen # Maximum Signal Length
        self.embed_dim = embed_dim # Input Embedding Dimensions

        #### Defining Layers
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen, 
            'embed_dim': self.embed_dim 
        })
        return config 

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

###### Encoder Block
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, dff_dim, rate=0.1):

        #### Defining Essentials
        super().__init__()
        self.d_model = d_model # Embedding Dimensions of the Encoder Layer
        self.num_heads = num_heads # Number of Self-Attention Heads
        self.dff_dim = dff_dim # Projection Dimensions of Feed-Forward Network
        self.rate = rate # Dropout Rate

        #### Defining Layers
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=self.d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff_dim, activation="relu"),
            tf.keras.layers.Dense(self.d_model),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model, 
            'num_heads': self.num_heads, 
            'dff_dim': self.dff_dim,
            'rate': self.rate
        })
        return config 

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # self-attention layer
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # layer norm
        ffn_output = self.ffn(out1)  #feed-forward layer
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # layer norm