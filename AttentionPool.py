####### Importing Libraries
import numpy as np
import tensorflow as tf

num_heads = 4
d_in = 32
thw_shape = [8,3,3]

def attn_pool(tensor, pool, thw_shape, norm):

    """
    Function to perform pooling

    INPUTS:-
    1) tensor: Input tensor of shape (B,num_heads,L,d_in/num_heads)
    2) pool: The pooling operation
    3) thw_shape: Current dimensions [T,H,W]
    4) norm: The norm operation

    OUTPUTS:-
    1) tensor: Output tensor of shape (B,num_heads,L_pool,d_in/num_heads)
    2) thw_shape: Dimensions after pooling [T,H,W]
    """

    B = tensor.shape[0] # Input batch size
    T, H, W = thw_shape # Input dimensions

    #tensor = tf.keras.layers.Reshape((self.num_heads,T,H,W,-1))(tensor)
    tensor = tf.reshape(tensor,
                            (-1,T,H,W,int(d_in //num_heads))
                            ) # shape -> (B*num_heads,T,H,W,d_in/num_heads)

    tensor = pool(tensor) # shape -> (B*num_heads,T_pool,H_pool,W_pool,d_in/num_heads)

    thw_shape = [tf.convert_to_tensor(tensor.shape[1],dtype=tf.int32),
                    tf.convert_to_tensor(tensor.shape[2],dtype=tf.int32),
                    tf.convert_to_tensor(tensor.shape[3],dtype=tf.int32)] # [T_pool,H_pool,W_pool]
    L_pool = tensor.shape[1]*tensor.shape[2]*tensor.shape[3] # Total tokens after pooling

    tensor = tf.reshape(tensor,(-1,num_heads,L_pool,int(d_in // num_heads)))
    # shape -> (B,num_heads,L_pool,d_in/num_heads)

    tensor = norm(tensor) # shape -> (B,num_heads,L_pool,d_in/num_heads)
    return tensor, thw_shape

a = tf.random.normal((32,4,72,8))
conv_layer = tf.keras.layers.Conv3D(8,(1,1,1),(1,1,1))
norm_layer = tf.keras.layers.LayerNormalization()
t, t_shape = attn_pool(a,conv_layer,thw_shape,norm_layer)
print(t.shape, t_shape)
