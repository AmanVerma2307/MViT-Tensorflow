####### Importing Libraries
import numpy as np
import tensorflow as tf

###### MViT MHSA
class MViT_MHSA(tf.keras.layers.Layer):

    """
    MViT's MHSA: Multi-scale variant of MHSA
    Pooling is performed first, and then attention.
    """

    def __init__(self,
                 d_in,
                 d_out,
                 num_heads,
                 q_kernel,
                 q_stride,
                 kv_kernel,
                 kv_stride):
        
        ##### Defining essentials
        super().__init__()
        self.d_in = d_in # Input dimensions
        self.d_out = d_out # Model Embedding Dimensions: Soft Attention requires d_out // num_heads = 0
        self.num_heads = num_heads # Number of attention heads
        self.q_kernel = q_kernel # Pooling kernel for Query
        self.q_stride = q_stride # Stride for Query
        self.kv_kernel = kv_kernel # Pooling kernel for Keys and Value
        self.kv_stride = kv_stride # Stride for Keys and 
        #self.no_pool = no_pool # Boolean to demarcate a simple MHSA from a MViT MHSA

        ##### Defining layers
        #### Linear layers
        self.query_dense = tf.keras.layers.Dense(self.d_out)
        self.key_dense = tf.keras.layers.Dense(self.d_out)
        self.value_dense = tf.keras.layers.Dense(self.d_out)
        self.concat_dense = tf.keras.layers.Dense(self.d_out)

        #### Pooling layers
        #if(no_pool == False):

        ### Query pool
        self.query_pool = tf.keras.layers.Conv3D(filters=int(self.d_in // self.num_heads),
                                                kernel_size=self.q_kernel,
                                                strides=self.q_stride,
                                                padding='valid',
                                                )
        self.query_norm = tf.keras.layers.LayerNormalization()
        
        ### Key pool
        self.key_pool = tf.keras.layers.Conv3D(filters=int(self.d_in // self.num_heads),
                                                kernel_size=self.kv_kernel,
                                                strides=self.kv_stride,
                                                padding='valid'
                                                )
        self.key_norm = tf.keras.layers.LayerNormalization()

        ### Value pool
        self.value_pool = tf.keras.layers.Conv3D(filters=int(self.d_in // self.num_heads),
                                                kernel_size=self.kv_kernel,
                                                strides=self.kv_stride,
                                                padding='valid'
                                                )
        self.value_norm = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_in': self.d_in,
            'd_out': self.d_out, 
            'num_heads': self.num_heads,
            'q_kernel': self.q_kernel,
            'kv_kernel': self.kv_kernel,
            'q_stride': self.q_stride,
            'kv_stride': self.kv_stride
        })
        return config 

    def split_heads(self, inputs,dim):
        inputs = tf.keras.layers.Reshape((-1,self.num_heads,int(dim // self.num_heads)))(inputs)
        #inputs = tf.reshape(inputs,(batch_size,-1,self.num_heads, int(dim // self.num_heads)))
        return tf.transpose(inputs, perm=[0,2,1,3]) # shape -> (B,num_heads,L,(dim//num_heads))
    
    def scaled_dot_product_attention(self,q,k,v):
        matmul_qk = tf.matmul(q,k,transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)
        output = tf.matmul(attention_weights,v)
        return output

    def attn_pool(self, tensor, pool, thw_shape, norm):

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
                            (-1,T,H,W,int(self.d_in // self.num_heads))
                            ) # shape -> (B*num_heads,T,H,W,d_in/num_heads)
        
        tensor = pool(tensor) # shape -> (B*num_heads,T_pool,H_pool,W_pool,d_in/num_heads)

        thw_shape = [tf.convert_to_tensor(tensor.shape[1],dtype=tf.int32),
                     tf.convert_to_tensor(tensor.shape[2],dtype=tf.int32),
                     tf.convert_to_tensor(tensor.shape[3],dtype=tf.int32)] # [T_pool,H_pool,W_pool]
        L_pool = tensor.shape[1]*tensor.shape[2]*tensor.shape[3] # Total tokens after pooling

        tensor = tf.reshape(tensor,(-1,self.num_heads,L_pool,int(self.d_in // self.num_heads)))
        # shape -> (B,num_heads,L_pool,d_in/num_heads)

        tensor = norm(tensor) # shape -> (B,num_heads,L_pool,d_in/num_heads)
        return tensor, thw_shape

    def call(self, x, thw_shape):

        """
        MViT MHSA

        INPUTS:-
        1) x: Input of shape (B,L,d_in)
        2) thw_shape: Original input dimensions [T,H,W]

        OUTPUTS:-
        1) x: Output of shape (B,L_q,d_out)
        2) q_shape: Output's video shape [T_q,H_q,W_q]
        """
        
        ##### Splitting heads
        B, L, _ = x.shape
        x = self.split_heads(x,self.d_in) # shape(x) -> (B,num_heads,L,d_in/num_heads)
        #print(x.shape)

        ##### Pooling Operation
        #if(self.no_pool == False):
        q, q_shape = self.attn_pool(x,
                                    self.query_pool,
                                    thw_shape,
                                    self.query_norm) # shape(q) -> (B,num_heads,L_q,d_in/num_heads)
        k, k_shape = self.attn_pool(x,
                                    self.key_pool,
                                    thw_shape,
                                    self.key_norm) # shape(k) -> (B,num_heads,L_k,d_in/num_heads)
        v, v_shape = self.attn_pool(x,
                                    self.value_pool,
                                    thw_shape,
                                    self.value_norm) # shape(v) -> (B,num_heads,L_v,d_in/num_heads)
        
        #print(q.shape, k.shape, v.shape)

        ##### Output dimension calculation
        L_q = q_shape[0]*q_shape[1]*q_shape[2] # L_q=T_q*H_q*W_q
        L_k = k_shape[0]*k_shape[1]*k_shape[2] # L_k=T_k*H_k*W_k
        L_v = v_shape[0]*v_shape[1]*v_shape[2] # L_v=T_v*H_v*W_v

        ##### Linear layers
        #### Query
        q = tf.transpose(q,perm=[0,2,1,3]) # shape(q) -> (B,L_q,num_heads,d_in/num_heads)
        q = tf.reshape(q,(-1,L_q,self.d_in)) # shape(q) -> (B,L_q,d_in)
        q = self.query_dense(q) # shape(q) -> (B,L_q,d_out)
        q = self.split_heads(q,self.d_out) # shape(q) -> (B,num_heads,L_q,d_out//num_heads)

        #### Key
        k = tf.transpose(k,perm=[0,2,1,3]) # shape(k) -> (B,L_k,num_heads,d_in/num_heads)
        k = tf.reshape(k,(-1,L_k,self.d_in)) # shape(k) -> (B,L_k,d_in)
        k = self.query_dense(k) # shape(k) -> (B,L_k,d_out)
        k = self.split_heads(k,self.d_out) # shape(k) -> (B,num_heads,L_k,d_out//num_heads)

        #### Value
        v = tf.transpose(v,perm=[0,2,1,3]) # shape(v) -> (B,L_v,num_heads,d_in/num_heads)
        v = tf.reshape(v,(-1,L_v,self.d_in)) # shape(v) -> (B,L_v,d_in)
        v = self.query_dense(v) # shape(v) -> (B,L_v,d_out)
        v = self.split_heads(v,self.d_out) # shape(v) -> (B,num_heads,L_v,d_out//num_heads)

        ##### Attention
        x = self.scaled_dot_product_attention(q,k,v) # shape(x) -> (B,num_heads,L_q,d_out//num_heads)
        
        ##### Residual connection
        x = x + q # shape(x) -> (B,num_heads,L_q,d_out//num_heads)

        ##### Projection layer
        x = tf.transpose(x,perm=[0,2,1,3]) # shape(x) -> (B,L_q,num_heads,d_out//num_heads)
        x = tf.reshape(x,(-1,L_q,self.d_out)) # shape(x) -> (B,L_q,d_out)
        x = self.concat_dense(x) # shape(x) -> (B,L_q,d_out)

        return x, q_shape
    
###### Testing
#a = tf.random.normal(shape=(25,320,32))
#b = tf.random.normal(shape=(25,10,128))
#thw_shape = [10,8,4]
#mvit_block_1 = MViT_MHSA(32,64,4,(2,2,2),(2,2,2),
#                          (1,1,1),(1,1,1))
#mvit_block_2 = MViT_MHSA(64,128,4,(1,2,2),
#                         (1,2,2),(1,1,1),(1,1,1))
#b, b_shape = mvit_block(a,thw_shape)
#print(b.shape, b_shape)

#Input_layer = tf.keras.layers.Input(shape=(320,32))
#output_layer, q_shape_1 = mvit_block_1(Input_layer,thw_shape)
#doutput_layer, q_shape_2 = mvit_block_2(output_layer,q_shape_1)
#model = tf.keras.models.Model(inputs=Input_layer,outputs=output_layer)
#model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='mse')
#model.summary()
#model.fit(a,b,epochs=100,batch_size=32)