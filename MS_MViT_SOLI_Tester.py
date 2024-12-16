####### Importing Libraries
import os
import time
import itertools
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from MViT_Encoder import MViT_Encoder
from ViViT import Tubelet_Embedding, PositionEmbedding, Encoder
from ICGD import ICGD_Loss

plt.switch_backend('agg')

####### Loading Dataset
###### Loading Arrays
X_train = np.load('./Datasets/SOLI/Metrics/Delta Distance/X_train_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']
X_dev = np.load('./Datasets/SOLI/Metrics/Delta Distance/X_dev_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']
y_train = np.load('./Datasets/SOLI/Metrics/Delta Distance/y_train_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/SOLI/Metrics/Delta Distance/y_dev_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']
y_train_id = np.load('./Datasets/SOLI/Metrics/Delta Distance/y_train_id_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./Datasets/SOLI/Metrics/Delta Distance/y_dev_id_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']

X_dev_nonshuffled = np.load('./Datasets/SOLI/DGBQA-Seen/X_dev_Seen-IAR-NonShuffled_SOLI.npz',allow_pickle=True)['arr_0']

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument("--lambda_id",
                    type=float,
                    help="Scaling Value of ID Loss")
parser.add_argument("--lambda_cgid",
                    type=float,
                    help="Scaling Value of CGID Loss")
parser.add_argument("--exp_name",
                    type=str,
                    help="Name of the Experiment being run, will be used saving the model and correponding outputs")

args = parser.parse_args()

####### Model Training
###### Defining Layers and Model

###### Defining Essentials
T = 40
H = 32
W = 32
C_rdi = 4
num_layers = 2
d_model = 32
num_heads = 16
dff_dim = 128
p_t = 2
p_h = 4
p_w = 4
n_t = (((T - p_t)//p_t)+1)
n_h = (((H - p_h)//p_h)+1)
n_w = (((W - p_w)//p_w)+1)
max_seq_len = int(n_t*(n_h)*(n_w))
pe_input = n_t*n_h*n_w
expansion_ratio = 4
rate = 0.3

###### Defining Layers

##### Convolutional Layers0

#### Res3DNet
conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))

conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

##### ViViT

#### tokenization
tubelet_embedding_layer = Tubelet_Embedding(d_model,(p_t,p_h,p_w))
positional_embedding_encoder = PositionEmbedding(max_seq_len,d_model)

#### Stage-1
block_11 = MViT_Encoder(d_model,d_model*2,num_heads,(2,2,2),
                            (2,2,2),(3,3,3),(3,3,3),
                            rate=0.3,dff_dim=128)
block_12 = Encoder(d_model*2,num_heads,dff_dim,rate)

#### Stage-2
block_21 = MViT_Encoder(d_model*2,d_model*4,num_heads,(2,1,1),
                            (2,1,1),(1,1,1),(1,1,1),
                            rate=0.3,dff_dim=128*2)
block_22 = Encoder(d_model*4,num_heads,dff_dim,rate)

#### Stage-3
block_31 = MViT_Encoder(d_model*2,d_model*4,num_heads,(1,2,2),
                            (1,2,2),(1,1,1),(1,1,1),
                            rate=0.3,dff_dim=128)
block_32 = Encoder(d_model*4,num_heads,dff_dim,rate)

#enc_block_1 = Encoder(d_model,num_heads,dff_dim,rate)
#enc_block_2 = Encoder(d_model,num_heads,dff_dim,rate)
#mvit_block_1 = MViT_Encoder(d_model,d_model,4,(2,2,2),
#                            (2,2,2),(1,1,1),(1,1,1),
#                            rate=0.3,dff_dim=128)
#mvit_block_2 = MViT_Encoder(d_model,d_model,4,(1,1,1),
#                            (1,1,1),(1,1,1),(1,1,1),
#                            rate=0.3,dff_dim=128)

###### Defining Model

##### Input Layer
Input_Layer = tf.keras.layers.Input(shape=(T,H,W,C_rdi))

##### Conv Layers

#### Res3DNet
### Residual Block - 1
conv11_rdi = conv11_rdi(Input_Layer)
conv12_rdi = conv12_rdi(conv11_rdi)
conv13_rdi = conv13_rdi(conv12_rdi)
conv13_rdi = tf.keras.layers.Add()([conv13_rdi,conv11_rdi])
#conv13_rdi = maxpool_1(conv13_rdi)

### Residual Block - 2
conv21_rdi = conv21_rdi(conv13_rdi)
conv22_rdi = conv22_rdi(conv21_rdi)
conv23_rdi = conv23_rdi(conv22_rdi)
conv23_rdi = tf.keras.layers.Add()([conv23_rdi,conv21_rdi])
#conv23_rdi = maxpool_2(conv23_rdi)

#####  ViViT
#### Embedding layers
tubelet_embedding = tubelet_embedding_layer(conv23_rdi)
tokens = positional_embedding_encoder(tubelet_embedding)
#enc_block_1_op = enc_block_1(tokens)
#enc_block_2_op, q_shape_1 = mvit_block_1(enc_block_1_op,[8,3,3])
#enc_block_3_op, q_shape_2 = mvit_block_2(enc_block_2_op,q_shape_1)

### Stage-1
block_11_op, block_11_shape = block_11(tokens,[n_t,n_h,n_w])
block_12_op = block_12(block_11_op)

### Stage-2
block_21_op, block_21_shape = block_21(block_12_op,block_11_shape)
block_22_op = block_22(block_21_op)

### Stage-3
#block_31_op, block_31_shape = block_31(block_22_op,block_21_shape)
#block_32_op = block_32(block_31_op)

##### Output Layer
gap_op = tf.keras.layers.GlobalAveragePooling1D()(block_22_op)
dense1 = tf.keras.layers.Dense(32,activation='relu')(gap_op)

#### HGR Output
dense2_hgr = tf.keras.layers.Dense(11,activation='softmax')(dense1)

#### ID Output
dense2_id = tf.keras.layers.Dense(10,activation='softmax')(dense1)

###### Compiling Model
model = tf.keras.models.Model(inputs=Input_Layer,outputs=[dense2_hgr,dense2_id,dense1])
model.load_weights('./Models/'+args.exp_name+'.h5')
model.summary()

####### Model Evaluation

##### Normalization Layer
def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

###### Extracting Model Outputs
g_hgr, g_id, f_theta = model.predict(X_dev)
f_theta_norm = tf.keras.layers.Lambda(normalisation_layer)(f_theta)
f_theta_norm = f_theta_norm.numpy()
G_bar = np.matmul(f_theta_norm,f_theta_norm.T) # Gram-Matrix

g_hgr_nonshuffled, g_id_nonshuffled, f_theta_nonshuffled = model.predict(X_dev_nonshuffled)
f_theta_nonshuffled_norm = tf.keras.layers.Lambda(normalisation_layer)(f_theta_nonshuffled)
f_theta_nonshuffled_norm = f_theta_nonshuffled_norm.numpy()
G_bar_nonshuffled = np.matmul(f_theta_nonshuffled_norm,f_theta_nonshuffled_norm.T) # Gram-Matrix

###### Accuracy Computations
##### Function to compute accuracy
def compute_accuracy(y_true,y_preds):
    
    """
    Function to compute accuracy in Sparse Categorical Prediction Style

    INPUTS:-
    1) y_true: Ground-trith sparse-categorical labels
    2) y_preds: Softmax layer outputs

    OUTPUTS:-
    1) acc_val: Accuracy Score
    """
    acc_val = tf.math.reduce_sum(tf.metrics.sparse_categorical_accuracy(y_true,y_preds))/y_true.shape[0]
    return acc_val

###### Accuracy Value
print('HGR Acc: '+str(compute_accuracy(y_dev,g_hgr))) # HGR Accuracy
print('ID Acc: '+str(compute_accuracy(y_dev_id,g_id))) # ID Accuracy

###### Saving Predictions
np.savez_compressed('./Predictions/'+args.exp_name+'.npz',g_hgr) # Saving Softmax predictions of shape: (N,G)

###### Softmax Heatmap
###### Computing Avg. Probability Scores
y_preds_hgr_probs = np.zeros((11,11))

##### Iterating over the Predicted Probabilites
for g_idx in range(11):

    g_prob = [] # List to store the Predicted Probabilties of the Current Class

    for idx, y_val_idx in enumerate(y_dev): # Iterating over the Dataset
        
        if(y_val_idx == g_idx):
            g_prob.append(g_hgr[idx])

    g_prob = np.around(np.mean(np.array(g_prob),axis=0),decimals=2)
    y_preds_hgr_probs[g_idx,:] = g_prob

print(y_preds_hgr_probs)

##### Saving Results
result_file = open('./Result Files/'+args.exp_name+'.txt','w')
result_file.write('HGR Acc: '+str(compute_accuracy(y_dev,g_hgr))+"\n")
result_file.write('ID Acc: '+str(compute_accuracy(y_dev_id,g_id))+"\n")
result_file.write(str(y_preds_hgr_probs))
result_file.close()

##### Plotting Heatmap

#### Heatmap Plotting Function
plt.rcParams["figure.figsize"] = [8,12]
def plot_heatmap(cm,filepath,classes,normalize=False,title='Avg. HGR Probabilities',cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_GramMatrix(cm,filepath,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

#### Heatmap Plotting
filepath='./Graphs/Softmax Heatmap/'+args.exp_name+'.png'
filepath_gram ='./Graphs/Gram Matrix/'+args.exp_name+'.png'
filepath_gram_ns ='./Graphs/Gram Matrix/'+args.exp_name+'_NonShuffled.png'
cm_plot_labels = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
plot_heatmap(cm=np.around(y_preds_hgr_probs,2),filepath=filepath,classes=cm_plot_labels,normalize=False,title='Avg. Softmax Probability')
plot_GramMatrix(cm=G_bar,filepath=filepath_gram)
plot_GramMatrix(cm=G_bar_nonshuffled,filepath=filepath_gram_ns)

###### tSNE

##### Embedding Function
col_mean = np.nanmean(f_theta_norm, axis=0)
inds = np.where(np.isnan(f_theta_norm))
#print(inds)
f_theta_norm[inds] = np.take(col_mean, inds[1])

##### Saving Embeddings
np.savez_compressed('./Embeddings/'+args.exp_name+'.npz',f_theta_norm)

##### t-SNE Plots
#### t-SNE Embeddings
tsne_X_dev = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(f_theta_norm) # t-SNE Plots 

#### Plotting
plt.rcParams["figure.figsize"] = [12,8]
for idx,color_index in zip(list(np.arange(11)),["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]):
    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='h')
plt.legend(['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold'],loc='best',prop={'size': 12})
#plt.grid(b='True',which='both')
plt.savefig('./Graphs/tSNE/'+args.exp_name+'.png')
