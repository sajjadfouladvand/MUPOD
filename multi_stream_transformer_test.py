#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# WRITTEN BY Sajjad Fouladvand (sjjd.fouladvand@gmail.com) based on Google implementation of Transformer here: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


import time
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import math
import scipy.special as sc
import random
from sklearn import metrics
from keras import regularizers

min_epoch = 10
F1_idx = 10
#================= Read test performances and find the best model =================
validation_res=[]
with open("validation_results_twostreamT_thresholding.csv") as validation_results:
  next(validation_results)
  for line in validation_results:
      line_perf=line.replace(',\n','').split(',')
      line_perf = [float(i) for i in line_perf]  
      validation_res.append(line_perf)
validation_res_ar=np.array(validation_res)
best_validation_res_ar=validation_res_ar[np.argmax(validation_res_ar[:,F1_idx]),:]
best_model_number= int(best_validation_res_ar[0])
learning_rate= best_validation_res_ar[1]
num_layers=int(best_validation_res_ar[2])
num_heads = int(best_validation_res_ar[3])
BATCH_SIZE=int(best_validation_res_ar[4])
dropout_rate=best_validation_res_ar[5]
EPOCHS=int(best_validation_res_ar[6])
optimum_threshold=best_validation_res_ar[17]
regu_factor = best_validation_res_ar[18]
#===================================================
#===================================================
model_number = 3000

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
tf.keras.backend.clear_session()
#========== Reading data============================
random.seed(time.clock())
d_model=16
d_meds=10
d_diags=10
d_demogs=2
zero=0
num_time_steps = 120
num_classes=2
one=1
two=2

class ReadingData(object):
    def __init__(self, path_t="", path_l=""):#, path_s=""):
        
        self.data = []
        self.labels = []
        self.seqlen = []
        s=[]
        temp=[]
        #pdb.set_trace()
        with open(path_t) as f:
              if path_t == 'testing_demogs_shuffled_balanced.csv':
                next(f)
              for line in f:
                  d_temp=line.split(',')
                  d_temp=[float(x) for x in d_temp]
                  self.data.append(d_temp)
                  d_temp=[]
                  s=[]
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        #pdb.set_trace()
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        if len(batch_data) < batch_size:
            batch_data = batch_data + (self.data[0:(batch_size - len(batch_data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data#, batch_labels#, batch_seqlen
#========== End of Reading data=====================


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(meds, diags):
  seq=np.ones((meds.shape[0], num_time_steps, d_model))
  for i in range(meds.shape[0]):
    meds_non_zero_map = np.equal(meds[i],0).all(axis=1)
    diags_non_zero_map = np.equal(diags[i],0).all(axis=1)
    if np.where(meds_non_zero_map==True)[0].size !=0:
      meds_zero_start = np.where(meds_non_zero_map==True)[0][0]
    else:
      meds_zero_start = num_time_steps-1
    if np.where(diags_non_zero_map==True)[0].size != 0:
      diags_zero_start = np.where(diags_non_zero_map==True)[0][0]
    else:
      diags_zero_start =  num_time_steps-1
    seq[i][max(meds_zero_start, diags_zero_start):]=0
  seq=tf.math.reduce_sum(seq, axis=2)
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def scaled_dot_product_attention(v,k,q,mask):#medications_v, medications_k, medications_q, diagnoses_v, diagnoses_k, diagnoses_q, procedures_v, procedures_k, procedures_q, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  #d_diagnoses_k = tf.cast(tf.shape(diagnoses_k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  # pdb.set_trace()
  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  if mask is not None:
    attention_weights *= tf.transpose((1-mask), perm=[0,1,3,2])
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    #pdb.set_trace()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wmq = tf.keras.layers.Dense(d_model)
    self.wmk = tf.keras.layers.Dense(d_model)
    self.wmv = tf.keras.layers.Dense(d_model)
    
    self.wdq = tf.keras.layers.Dense(d_model)
    self.wdk = tf.keras.layers.Dense(d_model)
    self.wdv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)
  def split_heads(self, x, batch_size):

    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, medications, diagnoses, mask):
    batch_size = tf.shape(medications)[0]

    medications_q = self.wmq(medications)  # (batch_size, seq_len, d_model)
    medications_k = self.wmk(medications)  # (batch_size, seq_len, d_model)
    medications_v = self.wmv(medications)  # (batch_size, seq_len, d_model)

    diagnoses_q = self.wdq(diagnoses)  # (batch_size, seq_len, d_model)
    diagnoses_k = self.wdk(diagnoses)  # (batch_size, seq_len, d_model)
    diagnoses_v = self.wdv(diagnoses)  # (batch_size, seq_len, d_model)    
    
    # pdb.set_trace()
    medications_q = self.split_heads(medications_q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    medications_k = self.split_heads(medications_k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    medications_v = self.split_heads(medications_v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    diagnoses_q = self.split_heads(diagnoses_q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    diagnoses_k = self.split_heads(diagnoses_k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    diagnoses_v = self.split_heads(diagnoses_v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

  #=============================================================================================================================================  
  #======================= Calculating Z_MM: the attention weights and final outputs for Medications-Medications interaction. 
  #=============================================================================================================================================
    scaled_attention_MM, attention_weights_MM = scaled_dot_product_attention(medications_v, medications_k, medications_q, mask)
    scaled_attention_MM = tf.transpose(scaled_attention_MM, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention_MM = tf.reshape(scaled_attention_MM, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    output_MM = self.dense(concat_attention_MM)  # (batch_size, seq_len_q, d_model)

  #=============================================================================================================================================  
  #======================= Calculating Z_MD: the attention weights and final outputs for Medication-Diagnoses interaction. 
  #=============================================================================================================================================
    scaled_attention_MD, attention_weights_MD = scaled_dot_product_attention(diagnoses_v, diagnoses_k, medications_q, mask)
    scaled_attention_MD = tf.transpose(scaled_attention_MD, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention_MD = tf.reshape(scaled_attention_MD, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    output_MD = self.dense(concat_attention_MD)  # (batch_size, seq_len_q, d_model)
   scaled_attention_DD, attention_weights_DD = scaled_dot_product_attention(diagnoses_v, diagnoses_k, diagnoses_q, mask)
    #scaled_attention, attention_weights = scaled_dot_product_attention(
    #    q, k, v, mask)    
    scaled_attention_DD = tf.transpose(scaled_attention_DD, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention_DD = tf.reshape(scaled_attention_DD, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    output_DD = self.dense(concat_attention_DD)  # (batch_size, seq_len_q, d_model)

  #=============================================================================================================================================  
  #======================= Calculating Z_PP: the attention weights and final outputs for Procedures-Procedures interaction. 
  #=============================================================================================================================================
    return output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_DD, attention_weights_DD

def point_wise_feed_forward_network(d_model, dff):
  return  tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def point_wise_feed_forward_network_M(d_model, dff):
  return  tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    #pdb.set_trace()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
 
    self.M_hat = tf.keras.layers.Dense(num_time_steps * d_model)
    self.D_hat = tf.keras.layers.Dense(num_time_steps * d_model)



  def call(self, medications, diagnoses, training, mask):
    # pdb.set_trace()
    batch_size = tf.shape(medications)[0]
    #================ Representation layer: here I map input data (medications, diagnoses, and procedures) to their Q, K, and V===================================================

    output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_DD, attention_weights_DD = self.mha(medications,diagnoses, mask)  # (batch_size, input_seq_len, d_model)
    
    output_MM = self.dropout1(output_MM, training=training)
    output_MM_1 = self.layernorm1(medications + output_MM)  # (batch_size, input_seq_len, d_model)    
    ffn_output_MM_1 = self.ffn(output_MM_1)  # (batch_size, input_seq_len, d_model)
    ffn_output_MM_1 = self.dropout2(ffn_output_MM_1, training=training)
    output_MM_2 = self.layernorm2(output_MM_1 + ffn_output_MM_1)  # (batch_size, input_seq_len, d_model)

    average_MD= tf.divide(tf.math.add_n([medications, diagnoses]), 2)
    output_MD = self.dropout1(output_MD, training=training)
    output_MD_1 = self.layernorm1(average_MD + output_MD)  # (batch_size, input_seq_len, d_model)    
    ffn_output_MD_1 = self.ffn(output_MD_1)  # (batch_size, input_seq_len, d_model)
    ffn_output_MD_1 = self.dropout2(ffn_output_MD_1, training=training)
    output_MD_2 = self.layernorm2(output_MD_1 + ffn_output_MD_1)  # (batch_size, input_seq_len, d_model)


    output_DD = self.dropout1(output_DD, training=training)
    output_DD_1 = self.layernorm1(diagnoses + output_DD)  # (batch_size, input_seq_len, d_model)    
    ffn_output_DD_1 = self.ffn(output_DD_1)  # (batch_size, input_seq_len, d_model)
    ffn_output_DD_1 = self.dropout2(ffn_output_DD_1, training=training)
    output_DD_2 = self.layernorm2(output_DD_1 + ffn_output_DD_1)  # (batch_size, input_seq_len, d_model)

    #pdb.set_trace()

    concatenated_outputs_forM = tf.concat([output_MM_2, output_MD_2], axis=2)         
    concatenated_outputs_forD = tf.concat([output_MD_2, output_DD_2], axis=2)         

    dimension_after_reshape_for_dense = num_time_steps * (d_model + d_model)
    concatenated_outputs_forM_reshaped = tf.reshape(concatenated_outputs_forM, (batch_size, dimension_after_reshape_for_dense))
    medications_hat_reshaped = self.M_hat(concatenated_outputs_forM_reshaped)
    medications_hat =  tf.reshape(medications_hat_reshaped, (batch_size, num_time_steps, d_model))
    
    concatenated_outputs_forD_reshaped = tf.reshape(concatenated_outputs_forD, (batch_size, dimension_after_reshape_for_dense))
    diagnoses_hat_reshaped = self.D_hat(concatenated_outputs_forD_reshaped)
    diagnoses_hat =  tf.reshape(diagnoses_hat_reshaped, (batch_size, num_time_steps, d_model))    

    return medications_hat, diagnoses_hat

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               rate=0.1):
    super(Encoder, self).__init__()
    #pdb.set_trace()
    self.d_model = d_model
    self.num_layers = num_layers
    
    #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x_1, x_2, training, mask):
    # pdb.set_trace()
    seq_len = tf.shape(x_1)[1]

    x_1 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x_2 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x_1 += self.pos_encoding[:, :seq_len, :]
    x_2 += self.pos_encoding[:, :seq_len, :]

    x_1 = self.dropout(x_1, training=training)
    x_2 = self.dropout(x_2, training=training)

    for i in range(self.num_layers):
      x_1, x_2 = self.enc_layers[i](x_1, x_2, training, mask)
    return x_1, x_2 # (batch_size, input_seq_len, d_model)

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()
    # pdb.set_trace()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    self.medications_map = tf.keras.layers.Dense(d_model)
    self.diagnoses_map = tf.keras.layers.Dense(d_model)

    self.w5 = tf.keras.layers.Dense(num_classes, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
  
  def call(self, medications, diagnoses, demog_info, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    # pdb.set_trace()
    batch_size = tf.shape(medications)[0]
    medications = self.medications_map(medications)
    diagnoses = self.diagnoses_map(diagnoses)
    enc_output_1, enc_output_2 = self.encoder(medications, diagnoses, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    # pdb.set_trace()
    #================= Trying MLP in the last layer
    concated_outputs= tf.concat([enc_output_1, enc_output_2, demog_info], axis=2)
    concated_outputs_reshaped = tf.reshape(concated_outputs, (batch_size, num_time_steps * (d_model + d_model + d_demogs)))

    logits=self.w5(concated_outputs_reshaped)

    return logits#, attention_weights

dff = d_model


input_vocab_size = 8500# tokenizer_pt.vocab_size + 2
target_vocab_size = 8000# tokenizer_en.vocab_size + 2


optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def my_loss(real, pred):

    loss=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real)  #+ L1 REGULARIZATION  
    return tf.reduce_mean(loss)#tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real)


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

checkpoint_path =  "./checkpoints/trained_model_" +str(best_model_number) 
checkpoint = tf.train.Checkpoint(transformer=transformer)

c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=EPOCHS)#, ...)

if c_manager.latest_checkpoint:
    tf.print("-----------Restoring from {}-----------".format(
        c_manager.latest_checkpoint))
    checkpoint.restore(c_manager.latest_checkpoint)


test_filename_meds='testing_medications_represented.csv'
test_filename_diags='testing_diagnoses_represented.csv'
test_filename_demogs='testing_demogs_shuffled_balanced.csv'
testset_meds = ReadingData(path_t=test_filename_meds)#, path_l=test_labels_filename)#,path_s=test_lengths_filename)
testset_diags = ReadingData(path_t=test_filename_diags)#, path_l=test_labels_filename)#,path_s=test_lengths_filename)
test_demogs = ReadingData(path_t=test_filename_demogs)
#===== test meds
test_data = testset_meds.data
test_data_ar = np.array(test_data)
test_meds_enrolids= test_data_ar[:,0]
# pdb.set_trace()
test_data_ar_reshaped = np.reshape(test_data_ar[:,one:-5],(len(test_data_ar), num_time_steps, d_meds))   
testset_labels = test_data_ar[:,-5:-3]
medications_test = tf.convert_to_tensor(test_data_ar_reshaped, np.float32)
test_meds_enrolids = test_data_ar[:,0]

#==== test diags
test_data = testset_diags.data 
test_data_ar=np.array(test_data)
test_diags_enrolids= test_data_ar[:,0]
test_data_ar_reshaped=np.reshape(test_data_ar[:,one:-5],(len(test_data_ar), num_time_steps, d_diags ))   
diagnoses_test = tf.convert_to_tensor(test_data_ar_reshaped, np.float32)
test_diags_enrolids = test_data_ar[:,0]

#==== test demogs
test_data = test_demogs.data
test_data_ar=np.array(test_data)
test_demogs_enrolids= test_data_ar[:,0]
batch_x_ar_reshaped=np.reshape(test_data_ar[:,one:],(len(test_data_ar), num_time_steps, d_demogs+1))   
test_demog_info=tf.convert_to_tensor(batch_x_ar_reshaped[:,:,one:], np.float32)


if (sum( test_meds_enrolids != test_diags_enrolids ) != 0) or (sum(test_diags_enrolids != test_demogs_enrolids) != 0):
  print("Error: enrolids don't match")
  pdb.set_trace()


#===== Breaking test set to k sets
test_split_k = 9803          
diagnoses_test_split=tf.split(value=diagnoses_test, num_or_size_splits=test_split_k)
medications_test_split=tf.split(value=medications_test, num_or_size_splits=test_split_k)
test_demog_info_split=tf.split(value=test_demog_info, num_or_size_splits=test_split_k)
logits_test_all = list()
for i in range(len(diagnoses_test_split)):
  val_enc_mask = create_padding_mask(medications_test_split[i], diagnoses_test_split[i])
  logits_test = transformer(medications_test_split[i], diagnoses_test_split[i], test_demog_info_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
  logits_test_all.extend(logits_test)

predictions_test_soft=tf.nn.softmax(logits_test_all)
np.savetxt("restoring_best_model_logits_test_twostreamT_"+str(model_number)+".csv", logits_test_all, delimiter=",")
np.savetxt("restoring_best_model__softmax_test_"+str(model_number)+".csv", predictions_test_soft.numpy(), delimiter=",")

#============ Thresholding for test=================
probabilities_test_pos=predictions_test_soft.numpy()[:,0]

fpr, tpr, thresholds = metrics.roc_curve(testset_labels[:,0], probabilities_test_pos, pos_label=1)
test_auc = metrics.auc(fpr, tpr)
tp_test=0
tn_test=0
fp_test=0
fn_test=0
for i in range(len(probabilities_test_pos)):      
    if(probabilities_test_pos[i]<optimum_threshold and testset_labels[i,0]==0):
        tn_test=tn_test+1
    elif(probabilities_test_pos[i]>=optimum_threshold and testset_labels[i,0]==1):
        tp_test=tp_test+1
    elif(probabilities_test_pos[i]>=optimum_threshold and testset_labels[i,0]==0):
        fp_test=fp_test+1
    elif(probabilities_test_pos[i]<optimum_threshold and testset_labels[i,0]==1):
        fn_test=fn_test+1
if((tp_test+fp_test)==0):
    precision_test=0
else:
    precision_test=tp_test/(tp_test+fp_test)
recall_test=tp_test/(tp_test+fn_test)
sensitivity_test=tp_test/(tp_test+fn_test)
specificity_test=tn_test/(tn_test+fp_test)    
if (precision_test+recall_test) !=0:
    F1Score_test=(2*precision_test*recall_test)/(precision_test+recall_test)      
else:
    F1Score_test=0        
accuracy_test= (tp_test+tn_test)/(tp_test+tn_test+fp_test+fn_test)
# pdb.set_trace()

header_results_filename= "Model Number (TWO STREAM), Learning Rate, Number of Layeres, Batch Size, Dropout Rate, Number of EPOCHS, test Accuracy, test Precision, test Recall, test F1-Score, test Specificity, test TP, test TN , test FP, test FN, test auc, test optimum threshold, regularization factor \n"
print("=================== TWO STREAM MODEL======================")
#pdb.set_trace()
with open("restoring_best_model_test_results_twostreamT_thresholdin.csv", 'w') as results_file:
      results_file.write("".join(["".join(x) for x in header_results_filename]))  
      results_file.write(str(best_model_number))
      results_file.write(",")
      results_file.write(str(learning_rate))
      results_file.write(",")
      results_file.write(str(num_layers))
      results_file.write(",")
      results_file.write(str(BATCH_SIZE))
      results_file.write(",")
      results_file.write(str(dropout_rate))
      results_file.write(",")
      results_file.write(str(EPOCHS))
      results_file.write(",")
      results_file.write(str(accuracy_test))
      results_file.write(", ")
      results_file.write(str(precision_test))
      results_file.write(", ")
      results_file.write(str(recall_test))
      results_file.write(", ")
      results_file.write(str(F1Score_test))
      results_file.write(", ")
      results_file.write(str(specificity_test))
      results_file.write(", ")                     
      results_file.write(str(tp_test))
      results_file.write(", ")
      results_file.write(str(tn_test))
      results_file.write(", ")
      results_file.write(str(fp_test))
      results_file.write(", ")
      results_file.write(str(fn_test))
      results_file.write(",")
      results_file.write(str(test_auc))
      results_file.write(",")
      results_file.write(str(optimum_threshold))
      results_file.write(",")
      results_file.write(str(regu_factor))
      results_file.write("\n")  
