###### Importing Libraries
import numpy as np
import tensorflow as tf

####### Patch Embedding Layer
class PatchEmbedding(tf.keras.layers.Layer):

    def __init__(self, T, embed_dim, patch_size):

        #### Defining Essentials
        super().__init__()
        self.T = T # Number of Frames
        self.embed_dim = embed_dim # Embedding Dimensions 
        self.patch_size = patch_size # A tuple of dimensions - (p_t,p_h,p_w), with each corresponding to patch dimensions

        #### Defining Layers
        self.embedding_layer = tf.keras.layers.Conv3D(filters=self.embed_dim,
                                                        kernel_size=self.patch_size,
                                                        strides=self.patch_size,
                                                        padding="VALID") # Tubelet Patch and Embedding Creation Layer
        self.flatten = tf.keras.layers.Reshape((-1,self.embed_dim)) # Layer to Flatten the Patches to Dimension (ST,D)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
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