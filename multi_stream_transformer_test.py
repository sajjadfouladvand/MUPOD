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
# pdb.set_trace()
validation_res=[]
with open("validation_results_twostreamT_thresholding.csv") as validation_results:
  next(validation_results)
  for line in validation_results:
      line_perf=line.replace(',\n','').split(',')
      #pdb.set_trace()
      # line_perf[17].replace('\n','')
      line_perf = [float(i) for i in line_perf]  
      validation_res.append(line_perf)
# pdb.set_trace()
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

#============== Read best model validation performances for early stopping
# pdb.set_trace()
# with open("validation_iterative_performance_thresholding_"+str(best_model_number)+".csv") as val_itr_perf_file:
#   next(val_itr_perf_file)
#   F1_scores_itr = []
#   for line in val_itr_perf_file:
#       F1_scores_itr.append(float(line.split(',')[3]))
# # pdb.set_trace()
# min_loss_idx = np.array(F1_scores_itr).argsort()[:EPOCHS]
# # pdb.set_trace()
# optimum_epoch = min_loss_idx[np.where(min_loss_idx >= min_epoch)[0][-1]] + 1
# pdb.set_trace()

# #============== Read validation losses and perform early stopping
# # pdb.set_trace()
# best_model_validation_loss = []
# with open("validation_loss_twostreamT_thresholding"+str(best_model_number)+".csv") as val_loss_file:
#   for line in val_loss_file:
#       best_model_validation_loss.append(float(line))
# # pdb.set_trace()
# min_loss_idx = np.array(best_model_validation_loss).argsort()[:EPOCHS]
# # pdb.set_trace()
# optimum_epoch = min_loss_idx[np.where(min_loss_idx >= min_epoch)[0][0]]

#import matplotlib.pyplot as plt
#CUDA_VISIBLE_DEVICES=3
#tf.enable_eager_execution() 
# pdb.set_trace()
#tf.compat.v1.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.keras.backend.clear_session()
# with open('error_log','w') as err_file:
#   err_file.write("List of functions...\n")
#===================================================
#===================================================
#========== Reading data============================
random.seed(time.clock())
d_model=16
d_meds=10
d_diags=10
d_demogs=2
#medication_dim=12
#diagnoses_dim=20
zero=0


#BATCH_SIZE = 128
#EPOCHS = 2
num_time_steps = 120
#num_inputs = 12
#x_dimension=84
#y_dimension=12
num_classes=2
one=1
two=2
#model_dimension=4
#num_layers = 4
#num_outputs=3
# num_input_mods=2
# d_model=12

#seq_max_len = 1176
# num_thresholds=100

# learning_rate_pool =[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]#[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.08, 0.1]
# EPOCHS_pool= [20, 40, 80, 120, 150]#, 150, 200]#, 110]#50, 100]#10, 15, 20]#[10000, 100000, 1000000, 2000000, 5000000, 10000000, 15000000]
# num_layers_pool=[1, 2, 4, 8]#[10, 20, 40, 60, 80, 100, 140, 180, 200, 300, 600]
# BATCH_SIZE_pool=[64, 128, 256, 512]#, 1024]
# dropout_rate_pool=[0.3, 0.4, 0.5, 0.6]
# # early_stopping_threshold_pool=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
# regularization_factor_pool = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]#, 0.01]
# num_heads_pool = [1,2,4,8]
# # pdb.set_trace()
# #======================= Generating parameters randomely
# learning_rate=random.choice(learning_rate_pool)
# num_layers=random.choice(num_layers_pool)
# BATCH_SIZE=random.choice(BATCH_SIZE_pool)
# dropout_rate=random.choice(dropout_rate_pool)
# EPOCHS=random.choice(EPOCHS_pool)
# regu_factor =random.choice(regularization_factor_pool)
# num_heads = random.choice(num_heads_pool)
# early_stopping_threshold=random.choice(early_stopping_threshold_pool)
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
              # temp_counter = 0
              for line in f:
                  #============================================================
                  #============================================================
                  #============================================================
                  # WARNING: THIS LINE WILL SAMPLE THE DATA====================
                  #============================================================
                  #============================================================
                  #============================================================
                  # temp_counter += 1
                  # if temp_counter >1000:
                  #   break
                  #============================================================
                  #============================================================
                  #============================================================                    
                  d_temp=line.split(',')
                  d_temp=[float(x) for x in d_temp]
                  #for i in range(max_seq_len):
                  #    temp.append(float(d_temp[i]))
                  #    s.append(temp)
                  #    temp=[]
                  #pdb.set_trace()
                  #dsds=s
                  self.data.append(d_temp)
                  d_temp=[]
                  s=[]
        #pdb.set_trace()
        # d_temp=[]
        # #temp=[]
        # with open(path_l) as f:
        #       for line in f:
        #           d_temp=[]
        #           d_temp=line.split(',')
        #           d_temp=[float(x) for x in d_temp]
        #           #temp.append(d_temp)
        #           #temp.append(int(float(d_temp[1])))
        #           #temp.append(int(float(d_temp[2])))
        #           self.labels.append(d_temp)                  
        #           d_temp=[]
        #pdb.set_trace()
        #with open(path_s) as f:
        #      for line in f:
        #          d_temp=[]
        #          temp=[]
        #          d_temp=line.split(',')
        #          d_temp = [int(float(i)) for i in d_temp]
                  #pdb.set_trace()
        #          self.seqlen.append(d_temp)
        #pdb.set_trace()
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        #pdb.set_trace()
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        # batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  # batch_size, len(self.data))])
        #batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
        #                                          batch_size, len(self.data))])
        if len(batch_data) < batch_size:
            #pdb.set_trace()
            batch_data = batch_data + (self.data[0:(batch_size - len(batch_data))])
            # batch_labels = batch_labels + (self.labels[0:(batch_size - len(batch_labels))])
           # batch_seqlen = batch_seqlen + (self.seqlen[0:(batch_size - len(batch_seqlen))])        
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data#, batch_labels#, batch_seqlen
#========== End of Reading data=====================
#===================================================
#===================================================

# pdb.set_trace()
# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
#                                as_supervised=True)
# train_examples, val_examples = examples['train'], examples['validation']

# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

# tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

# sample_string = 'Transformer is awesome.'

# tokenized_string = tokenizer_en.encode(sample_string)
# print ('Tokenized string is {}'.format(tokenized_string))

# original_string = tokenizer_en.decode(tokenized_string)
# print ('The original string: {}'.format(original_string))

# assert original_string == sample_string

# for ts in tokenized_string:
#   print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

# def encode(lang1, lang2):
#   lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
#       lang1.numpy()) + [tokenizer_pt.vocab_size+1]

#   lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
#       lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
#   return lang1, lang2

# MAX_LENGTH = 40

# def filter_max_length(x, y, max_length=MAX_LENGTH):
#   return tf.logical_and(tf.size(x) <= max_length,
#                     tf.size(y) <= max_length)
# def tf_encode(pt, en):
#   return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

# #pdb.set_trace()
# train_dataset = train_examples.map(tf_encode)
# train_dataset = train_dataset.filter(filter_max_length)
# # cache the dataset to memory to get a speedup while reading from it.
# train_dataset = train_dataset.cache()
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
#     BATCH_SIZE, padded_shapes=([-1], [-1]))
# train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# val_dataset = val_examples.map(tf_encode)
# val_dataset = val_dataset.filter(filter_max_length).padded_batch(
#     BATCH_SIZE, padded_shapes=([-1], [-1]))

# pt_batch, en_batch = next(iter(val_dataset))
# pt_batch, en_batch

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
#pos_encoding = positional_encoding(50, 8500)
#print (pos_encoding.shape)

# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, d_model))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
# def get_timing_signal_1d(length,
#                          channels,
#                          min_timescale=1.0,
#                          max_timescale=1.0e4,
#                          start_index=0):
#   """Gets a bunch of sinusoids of different frequencies.
#   Each channel of the input Tensor is incremented by a sinusoid of a different
#   frequency and phase.
#   This allows attention to learn to use absolute and relative positions.
#   Timing signals should be added to some precursors of both the query and the
#   memory inputs to attention.
#   The use of relative position is possible because sin(x+y) and cos(x+y) can be
#   expressed in terms of y, sin(x) and cos(x).
#   In particular, we use a geometric sequence of timescales starting with
#   min_timescale and ending with max_timescale.  The number of different
#   timescales is equal to channels / 2. For each timescale, we
#   generate the two sinusoidal signals sin(timestep/timescale) and
#   cos(timestep/timescale).  All of these sinusoids are concatenated in
#   the channels dimension.
#   Args:
#     length: scalar, length of timing signal sequence.
#     channels: scalar, size of timing embeddings to create. The number of
#         different timescales is equal to channels / 2.
#     min_timescale: a float
#     max_timescale: a float
#     start_index: index of first position
#   Returns:
#     a Tensor of timing signals [1, length, channels]
#   """
#   #pdb.set_trace()
#   position = tf.to_float(tf.range(length) + start_index)
#   num_timescales = channels // 2
#   log_timescale_increment = (
#       math.log(float(max_timescale) / float(min_timescale)) /
#       tf.maximum(tf.to_float(num_timescales) - 1, 1))
#   inv_timescales = min_timescale * tf.exp(
#       tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
#   scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
#   signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
#   signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
#   signal = tf.reshape(signal, [1, length, channels])
#   return signal
# pdb.set_trace()

#def create_padding_mask(seq):
#  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
#  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

#x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
#create_padding_mask(x)

#def create_look_ahead_mask(size):
#  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#  return mask  # (seq_len, seq_len)

#x = tf.random.uniform((1, 3))
#temp = create_look_ahead_mask(x.shape[1])
#temp


def create_padding_mask(meds, diags):
  # pdb.set_trace()
  #======== Find which stream has more records
  # np.sum(meds,axis=2)
  # with open('error_log','a') as err_file:
  #   err_file.write("Creating masks...\n")  
  seq=np.ones((meds.shape[0], num_time_steps, d_model))
  for i in range(meds.shape[0]):
    # print(i)
    # if i==32:
    # pdb.set_trace()
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
  # pdb.set_trace()
  seq=tf.math.reduce_sum(seq, axis=2)
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# def create_padding_mask_test(seq):
#   # pdb.set_trace()
#   seq=tf.math.reduce_sum(seq, axis=2)
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
#   # add extra dimensions to add the padding
#   # to the attention logits.
#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# pdb.set_trace()
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
  # pdb.set_trace()
  #medications_q_reshaped=tf.reshape(medications_q, [-1, model_dimension])
  #diagnoses_k_reshaped=tf.reshape(diagnoses_k, [-1, model_dimension])
  # if len(q)>128:
  #   pdb.set_trace()
  # pdb.set_trace()  
  #matmul_medications_q_diagnoses_k = tf.matmul(medications_q, diagnoses_k, transpose_b=True)

  # with open('error_log','a') as err_file:
  #   err_file.write("Scaled dot product attentio...\n")  
  # pdb.set_trace()
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
  # if len(attention_weights) ==458:
  #   pdb.set_trace()
  # print(attention_weights.shape)
  # print(mask.shape)
  # print("==================================")
  if mask is not None:
    attention_weights *= tf.transpose((1-mask), perm=[0,1,3,2])
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  #output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
# pdb.set_trace()
# def print_out(q, k, v):
#   temp_out, temp_attn = scaled_dot_product_attention(
#       q, k, v, None)
#   print ('Attention weights are:')
#   print (temp_attn)
#   print ('Output is:')
#   print (temp_out)

# np.set_printoptions(suppress=True)

# temp_k = tf.constant([[10,0,0],
#                       [0,10,0],
#                       [0,0,10],
#                       [0,0,10]], dtype=tf.float32)  # (4, 3)

# temp_v = tf.constant([[   1,0],
#                       [  10,0],
#                       [ 100,5],
#                       [1000,6]], dtype=tf.float32)  # (4, 2)

# # This `query` aligns with the second `key`,
# # so the second `value` is returned.
# temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
# print_out(temp_q, temp_k, temp_v)

# # This query aligns with a repeated key (third and fourth), 
# # so all associated values get averaged.
# temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
# print_out(temp_q, temp_k, temp_v)

# # This query aligns equally with the first and second key, 
# # so their values get averaged.
# temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
# print_out(temp_q, temp_k, temp_v)

# temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
# print_out(temp_q, temp_k, temp_v)

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
    # self.dense_outputs = tf.keras.layers.Dense(d_model)
  def split_heads(self, x, batch_size):
    # with open('error_log','a') as err_file:
    #   err_file.write("Split head...\n")    
    # print("")
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    #if x.shape[0] > 100:
    #  pdb.set_trace()
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, medications, diagnoses, mask):
    # pdb.set_trace()
    #batch_size = tf.shape(q)[0]
    # with open('error_log','a') as err_file:
    #   err_file.write("multi head att...\n")    
    batch_size = tf.shape(medications)[0]

    #q = self.wq(q)  # (batch_size, seq_len, d_model)
    #k = self.wk(k)  # (batch_size, seq_len, d_model)
    #v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    #q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    #k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    #v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

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

    #procedures_q = self.wq(procedures_q)  # (batch_size, seq_len, d_model)
    #procedures_k = self.wk(procedures_k)  # (batch_size, seq_len, d_model)
    #procedures_v = self.wv(procedures_v)  # (batch_size, seq_len, d_model)
    
    #procedures_q = self.split_heads(procedures_q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    #procedures_k = self.split_heads(procedures_k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    #procedures_v = self.split_heads(procedures_v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    #diagnoses_q = self.wq(diagnoses_q)  # (batch_size, seq_len, d_model)
    #diagnoses_k = self.wk(diagnoses_k)  # (batch_size, seq_len, d_model)
    #diagnoses_v = self.wv(diagnoses_v)  # (batch_size, seq_len, d_model)
    
    diagnoses_q = self.split_heads(diagnoses_q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    diagnoses_k = self.split_heads(diagnoses_k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    diagnoses_v = self.split_heads(diagnoses_v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)


  #=============================================================================================================================================  
  #======================= Calculating Z_MM: the attention weights and final outputs for Medications-Medications interaction. 
  #=============================================================================================================================================
    scaled_attention_MM, attention_weights_MM = scaled_dot_product_attention(medications_v, medications_k, medications_q, mask)
    #scaled_attention, attention_weights = scaled_dot_product_attention(
    #    q, k, v, mask)    
    scaled_attention_MM = tf.transpose(scaled_attention_MM, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention_MM = tf.reshape(scaled_attention_MM, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    output_MM = self.dense(concat_attention_MM)  # (batch_size, seq_len_q, d_model)

  #=============================================================================================================================================  
  #======================= Calculating Z_MD: the attention weights and final outputs for Medication-Diagnoses interaction. 
  #=============================================================================================================================================
    scaled_attention_MD, attention_weights_MD = scaled_dot_product_attention(diagnoses_v, diagnoses_k, medications_q, mask)
    #scaled_attention, attention_weights = scaled_dot_product_attention(
    #    q, k, v, mask)    
    scaled_attention_MD = tf.transpose(scaled_attention_MD, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention_MD = tf.reshape(scaled_attention_MD, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    output_MD = self.dense(concat_attention_MD)  # (batch_size, seq_len_q, d_model)

  #=============================================================================================================================================  
  #======================= Calculating Z_MP: the attention weights and final outputs for Medication-Procedures interaction. 
  #=============================================================================================================================================
    #scaled_attention_MP, attention_weights_MP = scaled_dot_product_attention(procedures_v, procedures_k, medications_q, mask)
    #scaled_attention, attention_weights = scaled_dot_product_attention(
    #    q, k, v, mask)    
    #scaled_attention_MP = tf.transpose(scaled_attention_MP, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    #concat_attention_MP = tf.reshape(scaled_attention_MP, 
    #                              (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    #output_MP = self.dense(concat_attention_MP)  # (batch_size, seq_len_q, d_model)

  #=============================================================================================================================================  
  #======================= Calculating Z_DP: the attention weights and final outputs for Diagnoses-Procedures interaction. 
  #=============================================================================================================================================
    #scaled_attention_DP, attention_weights_DP = scaled_dot_product_attention(procedures_v, procedures_k, diagnoses_q, mask)
    #scaled_attention, attention_weights = scaled_dot_product_attention(
    #    q, k, v, mask)    
    #scaled_attention_DP = tf.transpose(scaled_attention_DP, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    #concat_attention_DP = tf.reshape(scaled_attention_DP, 
    #                              (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    #output_DP = self.dense(concat_attention_DP)  # (batch_size, seq_len_q, d_model)

  #=============================================================================================================================================  
  #======================= Calculating Z_DD: the attention weights and final outputs for Diagnoses-Diagnoses interaction. 
  #=============================================================================================================================================
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
    #pdb.set_trace()
    #scaled_attention_PP, attention_weights_PP = scaled_dot_product_attention(procedures_v, procedures_k, procedures_q, mask)
    #scaled_attention, attention_weights = scaled_dot_product_attention(
    #    q, k, v, mask)    
    #scaled_attention_PP = tf.transpose(scaled_attention_PP, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    #concat_attention_PP = tf.reshape(scaled_attention_PP, 
    #                              (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    #output_PP = self.dense(concat_attention_PP)  # (batch_size, seq_len_q, d_model)
    #pdb.set_trace()
    

    #concat_all_attentions= tf.concat([concat_attention_MM, concat_attention_MD, concat_attention_MP, concat_attention_DP, concat_attention_DD, concat_attention_PP],axis=0)
    #output_all=self.dense(concat_all_attentions)
    #tf.split(output_all, 3)
    return output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_DD, attention_weights_DD

#temp_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
#y = tf.random.uniform((BATCH_SIZE, num_time_steps, d_model))  # (batch_size, encoder_sequence, d_model)

#output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_MP, attention_weights_MP, output_DP, attention_weights_DP, output_DD, attention_weights_DD, output_PP, attention_weights_PP = temp_mha(y, medications_k=y, medications_q=y, diagnoses_v=y, diagnoses_k=y, diagnoses_q=y, procedures_v=y, procedures_k=y, procedures_q=y, mask=None)

#out.shape, attn.shape
# pdb.set_trace()
def point_wise_feed_forward_network(d_model, dff):
  return  tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
#pdb.set_trace()
#sample_ffn = point_wise_feed_forward_network(d_model, 2048)
#sample_ffn(tf.random.uniform((64, 50, d_model))).shape


def point_wise_feed_forward_network_M(d_model, dff):
  return  tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
#sample_ffn = point_wise_feed_forward_network(d_model, 2048)
#sample_ffn(tf.random.uniform((64, 50, d_model))).shape
# pdb.set_trace()
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

    #============ Patients representation weights and biases
    #====================================================
    #self.W_medV = tf.Variable(tf.random.normal([d_model, d_model]), name="WmedV")
    #self.b_medV = tf.Variable(tf.random.normal([d_model]), name="BmedV")

    #self.W_medK = tf.Variable(tf.random.normal([d_model, d_model]), name="WmedK")
    #self.b_medK = tf.Variable(tf.random.normal([d_model]))

    #self.W_medQ = tf.Variable(tf.random.normal([d_model, d_model]), name="WmedQ")
    #self.b_medQ = tf.Variable(tf.random.normal([d_model]))

    #self.W_diagV = tf.Variable(tf.random.normal([d_model, d_model]), name="WdiagV")
    #self.b_diagV = tf.Variable(tf.random.normal([d_model]))

    #self.W_diagK = tf.Variable(tf.random.normal([d_model, d_model]), name="WdiagK")
    #self.b_diagK = tf.Variable(tf.random.normal([d_model]))

    #self.W_diagQ = tf.Variable(tf.random.normal([d_model, d_model]), name="WdiagQ")
    #self.b_diagQ = tf.Variable(tf.random.normal([d_model]))

    #self.W_procV = tf.Variable(tf.random.normal([medication_dim, medication_dim]), name="WprocV")
    #self.b_procV = tf.Variable(tf.random.normal([medication_dim]))

    #self.W_procK = tf.Variable(tf.random.normal([medication_dim, medication_dim]), name="WprocK")
    #self.b_procK = tf.Variable(tf.random.normal([medication_dim]))

    #self.W_procQ = tf.Variable(tf.random.normal([medication_dim, medication_dim]), name="WprocQ")
    #self.b_procQ = tf.Variable(tf.random.normal([medication_dim]))

    # Using shared matrixes for converting medications, diagnoses and procedure to q, v, and k matrixes
    #self.W_allV = tf.Variable(tf.random.normal([d_model, d_model]), name="WallV")
    #self.b_allV = tf.Variable(tf.random.normal([d_model]), name="BallV")

    #self.W_allK = tf.Variable(tf.random.normal([d_model, d_model]), name="WallK")
    #self.b_allK = tf.Variable(tf.random.normal([d_model]))

    #self.W_allQ = tf.Variable(tf.random.normal([d_model, d_model]), name="WallQ")
    #self.b_allQ = tf.Variable(tf.random.normal([d_model]))

    #====================================================
    #====================================================
    
    #=========== Weights and biases to turn back the out puts of the self attention to the three modalities of the data for the next encoder layer
    #self.W_M_hat = tf.Variable(tf.random.normal([d_model, d_model]), name="W_mHat")
    #self.b_M_hat = tf.Variable(tf.random.normal([d_model]))

    #self.W_D_hat = tf.Variable(tf.random.normal([d_model, d_model]), name="W_dHat")
    #self.b_D_hat = tf.Variable(tf.random.normal([d_model]))
    
    self.M_hat = tf.keras.layers.Dense(num_time_steps * d_model)
    self.D_hat = tf.keras.layers.Dense(num_time_steps * d_model)
    #self.W_P_hat = tf.Variable(tf.random.normal([d_model, d_model]), name="W_pHat")
    #self.b_P_hat = tf.Variable(tf.random.normal([d_model]))  



  def call(self, medications, diagnoses, training, mask):
    # pdb.set_trace()
    batch_size = tf.shape(medications)[0]
    #================ Representation layer: here I map input data (medications, diagnoses, and procedures) to their Q, K, and V===================================================
    #=============================================================================================================================================================================
    #medications_reshaped = tf.reshape(medications, [batch_size * num_time_steps, -1])
    #medications_v = tf.reshape(tf.matmul(medications_reshaped, self.W_medV) + self.b_medV, [batch_size, num_time_steps, d_model]) 
    #medications_k = tf.reshape(tf.matmul(medications_reshaped, self.W_medK) + self.b_medK, [batch_size, num_time_steps, d_model]) 
    #medications_q = tf.reshape(tf.matmul(medications_reshaped, self.W_medQ) + self.b_medQ, [batch_size, num_time_steps, d_model]) 

    #diagnoses_reshaped = tf.reshape(diagnoses, [batch_size * num_time_steps, -1])
    #diagnoses_v = tf.reshape(tf.matmul(diagnoses_reshaped, self.W_diagV) + self.b_diagV, [batch_size, num_time_steps, d_model]) 
    #diagnoses_k = tf.reshape(tf.matmul(diagnoses_reshaped, self.W_diagK) + self.b_diagK, [batch_size, num_time_steps, d_model]) 
    #diagnoses_q = tf.reshape(tf.matmul(diagnoses_reshaped, self.W_diagQ) + self.b_diagQ, [batch_size, num_time_steps, d_model]) 

    #procedures_reshaped = tf.reshape(procedures, [batch_size * num_time_steps, -1])
    #procedures_v = tf.reshape(tf.matmul(procedures_reshaped, self.W_allV) + self.b_allV, [batch_size, num_time_steps, d_model]) 
    #procedures_k = tf.reshape(tf.matmul(procedures_reshaped, self.W_allK) + self.b_allK, [batch_size, num_time_steps, d_model]) 
    #procedures_q = tf.reshape(tf.matmul(procedures_reshaped, self.W_allQ) + self.b_allQ, [batch_size, num_time_steps, d_model]) 

    #================End of representation layer==================================================================================================================================
    #=============================================================================================================================================================================
    #x_4=x_1     
    #x_5=x_1
    #x_6= x_1
    #x_7= x_1
    #x_8= x_1
    #x_9= x_1

    output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_DD, attention_weights_DD = self.mha(medications,diagnoses, mask)  # (batch_size, input_seq_len, d_model)
    
    #============ I can convert the output of multihead attention to three matrixes here

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

    #average_MP= tf.divide(tf.math.add_n([medications, procedures]), 2)
    #output_MP = self.dropout1(output_MP, training=training)
    #output_MP_1 = self.layernorm1(average_MP + output_MP)  # (batch_size, input_seq_len, d_model)    
    #ffn_output_MP_1 = self.ffn(output_MP_1)  # (batch_size, input_seq_len, d_model)
    #ffn_output_MP_1 = self.dropout2(ffn_output_MP_1, training=training)
    #output_MP_2 = self.layernorm2(output_MP_1 + ffn_output_MP_1)  # (batch_size, input_seq_len, d_model)

    #average_DP= tf.divide(tf.math.add_n([diagnoses, procedures]), 2)
    #output_DP = self.dropout1(output_DP, training=training)
    #output_DP_1 = self.layernorm1(average_DP + output_DP)  # (batch_size, input_seq_len, d_model)    
    #ffn_output_DP_1 = self.ffn(output_DP_1)  # (batch_size, input_seq_len, d_model)
    #ffn_output_DP_1 = self.dropout2(ffn_output_DP_1, training=training)
    #output_DP_2 = self.layernorm2(output_DP_1 + ffn_output_DP_1)  # (batch_size, input_seq_len, d_model)

    output_DD = self.dropout1(output_DD, training=training)
    output_DD_1 = self.layernorm1(diagnoses + output_DD)  # (batch_size, input_seq_len, d_model)    
    ffn_output_DD_1 = self.ffn(output_DD_1)  # (batch_size, input_seq_len, d_model)
    ffn_output_DD_1 = self.dropout2(ffn_output_DD_1, training=training)
    output_DD_2 = self.layernorm2(output_DD_1 + ffn_output_DD_1)  # (batch_size, input_seq_len, d_model)

    #output_PP = self.dropout1(output_PP, training=training)
    #output_PP_1 = self.layernorm1(procedures + output_PP)  # (batch_size, input_seq_len, d_model)    
    #ffn_output_PP_1 = self.ffn(output_PP_1)  # (batch_size, input_seq_len, d_model)
    #ffn_output_PP_1 = self.dropout2(ffn_output_PP_1, training=training)
    #output_PP_2 = self.layernorm2(output_PP_1 + ffn_output_PP_1)  # (batch_size, input_seq_len, d_model)
    

    #pdb.set_trace()
    #concatenated_outputs = tf.concat([output_MM_2, output_MD_2, output_MP_2, output_DP_2, output_DD_2, output_PP_2], axis=1)            
    concatenated_outputs_forM = tf.concat([output_MM_2, output_MD_2], axis=2)         
    #averaged_outputs_forM=tf.divide(tf.math.add_n([output_MM_2, output_MD_2]), num_input_mods)
    #averaged_outputs_forD=tf.divide(tf.math.add_n([output_MD_2, output_DD_2]), num_input_mods)
    #averaged_outputs_forP=tf.divide(tf.math.add_n([output_MP_2, output_DP_2, output_PP_2]), num_input_mods)
    concatenated_outputs_forD = tf.concat([output_MD_2, output_DD_2], axis=2)         
    #concatenated_outputs_forP = tf.concat([output_MP_2, output_DP_2, output_PP_2], axis=1)         
    
    #concatenated_outputs= self.ffn(concatenated_outputs)
    #concatenated_outputs_reshaped=tf.reshape(concatenated_outputs, [BATCH_SIZE * x_dimension * num_outputs,-1])
    #averaged_outputs_forM_reshaped = tf.reshape(averaged_outputs_forM, [batch_size * num_time_steps ,-1])
    #averaged_outputs_forD_reshaped = tf.reshape(averaged_outputs_forD, [batch_size * num_time_steps ,-1])
    #concatenated_outputs_forM_reshaped = tf.reshape(concatenated_outputs_forM, [batch_size * num_time_steps ,-1])
    #concatenated_outputs_forD_reshaped = tf.reshape(concatenated_outputs_forD, [batch_size * num_time_steps ,-1])    
    #averaged_outputs_forP_reshaped = tf.reshape(averaged_outputs_forP, [batch_size * num_time_steps ,-1])
    #pdb.set_trace()
    dimension_after_reshape_for_dense = num_time_steps * (d_model + d_model)
    concatenated_outputs_forM_reshaped = tf.reshape(concatenated_outputs_forM, (batch_size, dimension_after_reshape_for_dense))
    medications_hat_reshaped = self.M_hat(concatenated_outputs_forM_reshaped)
    medications_hat =  tf.reshape(medications_hat_reshaped, (batch_size, num_time_steps, d_model))
    
    concatenated_outputs_forD_reshaped = tf.reshape(concatenated_outputs_forD, (batch_size, dimension_after_reshape_for_dense))
    diagnoses_hat_reshaped = self.D_hat(concatenated_outputs_forD_reshaped)
    diagnoses_hat =  tf.reshape(diagnoses_hat_reshaped, (batch_size, num_time_steps, d_model))    
    #diagnoses_hat = tf.reshape(self.M_hat(tf.reshape(concatenated_outputs_forD, (batch_size, dimension_after_reshape_for_dense))), (batch_size, num_time_steps, d_model))
    #medications_hat = self.M_hat(concatenated_outputs_forM)
    #diagnoses_hat = self.D_hat(concatenated_outputs_forD)

    #medications_hat=tf.matmul(averaged_outputs_forM_reshaped, self.W_M_hat) + self.b_M_hat
    #diagnoses_hat=tf.matmul(averaged_outputs_forD_reshaped, self.W_D_hat) + self.b_D_hat
    #procedures_hat=tf.matmul(averaged_outputs_forP_reshaped, self.W_P_hat) + self.b_P_hat
    #medications_hat=tf.reshape(medications_hat, [batch_size, num_time_steps, d_model])
    #diagnoses_hat=tf.reshape(diagnoses_hat, [batch_size, num_time_steps, d_model])
    #procedures_hat=tf.reshape(procedures_hat, [batch_size, num_time_steps, d_model])
    #pdb.set_trace()
    #num_outputs * np.shape(x_1)[1]
    #self.ffn(concatenated_outputs)
    #output_all_1 = 
    #output_all_2 = 
    #output_all_3 = 
    return medications_hat, diagnoses_hat
# pdb.set_trace()
#sample_encoder_layer = EncoderLayer(d_model, num_heads, 2048)
#pdb.set_trace()
#x_1, x_2, x_3 = sample_encoder_layer(
#    tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension)), tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension)), tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension)), False, None)
#pdb.set_trace()
#sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)

# class DecoderLayer(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads, dff, rate=0.1):
#     super(DecoderLayer, self).__init__()
#     #pdb.set_trace()
#     self.mha1 = MultiHeadAttention(d_model, num_heads)
#     self.mha2 = MultiHeadAttention(d_model, num_heads)

#     self.ffn = point_wise_feed_forward_network(d_model, dff)
 
#     self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
#     self.dropout1 = tf.keras.layers.Dropout(rate)
#     self.dropout2 = tf.keras.layers.Dropout(rate)
#     self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
#   def call(self, x, enc_output, training, 
#            look_ahead_mask, padding_mask):
#     # enc_output.shape == (batch_size, input_seq_len, d_model)
#     #pdb.set_trace()
#     attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#     attn1 = self.dropout1(attn1, training=training)
#     out1 = self.layernorm1(attn1 + x)
    
#     attn2, attn_weights_block2 = self.mha2(
#         enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
#     attn2 = self.dropout2(attn2, training=training)
#     out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
#     ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
#     ffn_output = self.dropout3(ffn_output, training=training)
#     out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
#     return out3, attn_weights_block1, attn_weights_block2

# sample_decoder_layer = DecoderLayer(d_model, num_heads, 2048)

# sample_decoder_layer_output, _, _ = sample_decoder_layer(
#     tf.random.uniform((64, 50, d_model)), sample_encoder_layer_output, 
#     False, None, None)

# sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)

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
    #x_1=tf.dtypes.cast(x_1, tf.float32)
    #x_2=tf.dtypes.cast(x_2, tf.float32)
    #x_3=tf.dtypes.cast(x_3, tf.float32)
    #np.savetxt("x_1_before_positional.csv", x_1.numpy())
    # adding embedding and position encoding.
    #x = self.embedding(x_1)  # (batch_size, input_seq_len, d_model)
    x_1 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x_2 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += self.pos_encoding[:, :seq_len, :]
    x_1 += self.pos_encoding[:, :seq_len, :]
    x_2 += self.pos_encoding[:, :seq_len, :]
    #x_3 += self.pos_encoding[:, :seq_len, :]
    x_1 = self.dropout(x_1, training=training)
    x_2 = self.dropout(x_2, training=training)
    #x_3 = self.dropout(x_3, training=training)
    for i in range(self.num_layers):
      x_1, x_2 = self.enc_layers[i](x_1, x_2, training, mask)
      #pdb.set_trace()
      #print("Here I should combine encoder layer outputs")
      #averaged_output = tf.math.divide(tf.math.add_n([output_MM_2, output_MD_2, output_MP_2, output_DP_2, output_DD_2, output_PP_2]), num_outputs)      

    return x_1, x_2 # (batch_size, input_seq_len, d_model)
# pdb.set_trace()
#sample_encoder = Encoder(num_layers=2, d_model=d_model, num_heads=num_heads, 
#                         dff=2048, input_vocab_size=8500)
#pdb.set_trace()
#x_1, x_2, x_3 = sample_encoder(tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension)),tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension)), tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension)),
#                                       training=False, mask=None)
#pdb.set_trace()
#print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
#pdb.set_trace()
# class Decoder(tf.keras.layers.Layer):
#   def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
#                rate=0.1):
#     super(Decoder, self).__init__()
#     #pdb.set_trace()
#     self.d_model = d_model
#     self.num_layers = num_layers
    
#     self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
#     self.pos_encoding = positional_encoding(target_vocab_size, d_model)
    
#     self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
#                        for _ in range(num_layers)]
#     self.dropout = tf.keras.layers.Dropout(rate)
    
#   def call(self, x, enc_output, training, 
#            look_ahead_mask, padding_mask):
#     #pdb.set_trace()
#     seq_len = tf.shape(x)[1]
#     attention_weights = {}
    
#     x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x += self.pos_encoding[:, :seq_len, :]
    
#     x = self.dropout(x, training=training)

#     for i in range(self.num_layers):
#       x, block1, block2 = self.dec_layers[i](x, enc_output, training,
#                                              look_ahead_mask, padding_mask)
      
#       attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
#       attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
#     # x.shape == (batch_size, target_seq_len, d_model)
#     return x, attention_weights

# sample_decoder = Decoder(num_layers=2, d_model=d_model, num_heads=num_heads, 
#                          dff=2048, target_vocab_size=8000)

# output, attn = sample_decoder(tf.random.uniform((64, 26)), 
#                               enc_output=sample_encoder_output, 
#                               training=False, look_ahead_mask=None, 
#                               padding_mask=None)

# output.shape, attn['decoder_layer2_block2'].shape

# pdb.set_trace()
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()
    # pdb.set_trace()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    #self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
    #                       target_vocab_size, rate)

    #self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    self.medications_map = tf.keras.layers.Dense(d_model)
    self.diagnoses_map = tf.keras.layers.Dense(d_model)
    
    
    ####self.W_final = tf.Variable(tf.random.normal([ num_time_steps * d_model, num_classes]), name="W_final")
    ####self.b_final = tf.Variable(tf.random.normal([num_classes]), name="B_final")
    
    #===== Weight and biases for MLP
    # self.w1 = tf.keras.layers.Dense(d_model *2, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
    # self.w2 = tf.keras.layers.Dense(d_model * 4, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
    # self.w3 = tf.keras.layers.Dense(d_model *2, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
    # self.w4 = tf.keras.layers.Dense(int(d_model/2), kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
    self.w5 = tf.keras.layers.Dense(num_classes, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
    # self.ckpt = self.makeCheckpoint()
    # self.W_inp = tf.Variable(tf.random.normal([ num_time_steps * (d_model + d_model), 1024]), name="W_inp")
    # self.b_inp = tf.Variable(tf.random.normal([1024]), name="B_inp")
    # self.W_1 = tf.Variable(tf.random.normal([ 1024, 512]), name="W_1")
    # self.b_1 = tf.Variable(tf.random.normal([512]), name="B_1")    
    # self.W_2 = tf.Variable(tf.random.normal([ 512, 256]), name="W_2")
    # self.b_2 = tf.Variable(tf.random.normal([256]), name="B_2")    
    # self.W_3 = tf.Variable(tf.random.normal([ 256, 128]), name="W_3")
    # self.b_3 = tf.Variable(tf.random.normal([128]), name="B_3")    
    # self.W_4 = tf.Variable(tf.random.normal([ 128, 64]), name="W_4")
    # self.b_4 = tf.Variable(tf.random.normal([64]), name="B_4")    
    # self.W_5 = tf.Variable(tf.random.normal([ 64, 32]), name="W_5")
    # self.b_5 = tf.Variable(tf.random.normal([32]), name="B_5")    
    # self.W_6 = tf.Variable(tf.random.normal([ 32, 16]), name="W_6")
    # self.b_6 = tf.Variable(tf.random.normal([16]), name="B_6")       
    # self.W_7 = tf.Variable(tf.random.normal([ 16, num_classes]), name="W_7")
    # self.b_7 = tf.Variable(tf.random.normal([num_classes]), name="B_7")                  
    #===== Weight and biases for MLP
  # def makeCheckpoint(self):
  #   return tf.train.Checkpoint(
  #       encoder=self.encoder, medications_map = self.medications_map,
  #       diagnoses_map = self.diagnoses_map, w5=self.w5)  
  # def saveVariables(self):
  #   self.ckpt.save('./ckpt')  
  # def restoreVariables(self):
  #   status = self.ckpt.restore(tf.train.latest_checkpoint('.'))
  #   status.assert_consumed()  # Optional check      
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
    # concated_outputs_reshaped=self.w1(concated_outputs_reshaped)
    # concated_outputs_reshaped=self.w2(concated_outputs_reshaped)
    # concated_outputs_reshaped=self.w3(concated_outputs_reshaped)
    # concated_outputs_reshaped=self.w4(concated_outputs_reshaped)
    logits=self.w5(concated_outputs_reshaped)
    # hidden_inp_layer_vals = {'weights':self.W_inp,'biases':self.b_inp}
    # output_layer_inp = tf.nn.sigmoid(tf.add(tf.matmul(concated_outputs_reshaped,hidden_inp_layer_vals['weights']),hidden_inp_layer_vals['biases']))

    # hidden_1_layer_vals = {'weights':self.W_1,'biases':self.b_1}
    # output_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_inp,hidden_1_layer_vals['weights']),hidden_1_layer_vals['biases']))

    # hidden_2_layer_vals = {'weights':self.W_2,'biases':self.b_2}
    # output_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_1,hidden_2_layer_vals['weights']),hidden_2_layer_vals['biases']))

    # hidden_3_layer_vals = {'weights':self.W_3,'biases':self.b_3}
    # output_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_2,hidden_3_layer_vals['weights']),hidden_3_layer_vals['biases']))

    # hidden_4_layer_vals = {'weights':self.W_4,'biases':self.b_4}
    # output_layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_3,hidden_4_layer_vals['weights']),hidden_4_layer_vals['biases']))

    # hidden_5_layer_vals = {'weights':self.W_5,'biases':self.b_5}
    # output_layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_4,hidden_5_layer_vals['weights']),hidden_5_layer_vals['biases']))

    # hidden_6_layer_vals = {'weights':self.W_6,'biases':self.b_6}
    # output_layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_5,hidden_6_layer_vals['weights']),hidden_6_layer_vals['biases']))

    # hidden_7_layer_vals = {'weights':self.W_7,'biases':self.b_7}
    # logits = tf.nn.sigmoid(tf.add(tf.matmul(output_layer_6,hidden_7_layer_vals['weights']),hidden_7_layer_vals['biases']))    
    #predictions =  tf.nn.softmax(logits)
    #================= END OF Trying MLP in the last layer    
    ####averaged_enc_outputs = tf.divide(tf.math.add_n([enc_output_1, enc_output_2]), num_input_mods)    
    ####averaged_enc_outputs_reshaped = tf.reshape(averaged_enc_outputs, (batch_size, num_time_steps * d_model))
    #enc_outputs_concated=tf.reshape(tf.concat([enc_output_1, enc_output_1, enc_output_1], axis=1), (batch_size , num_input_mods * x_dimension * y_dimension))
    ####logits=tf.matmul( averaged_enc_outputs_reshaped, self.W_final) + self.b_final
    ####predictions =  tf.nn.softmax(logits)
    #predictions =  tf.math.tanh(tf.matmul( averaged_enc_outputs_reshaped, self.W_final) + self.b_final)
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    #dec_output, attention_weights = self.decoder(
    #    tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    #final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return logits#, attention_weights

# sample_transformer = Transformer(
#     num_layers=2, d_model=d_model, num_heads=num_heads, dff=2048, 
#     input_vocab_size=8500, target_vocab_size=8000)

# temp_input = tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension))
# temp_target = tf.random.uniform((BATCH_SIZE, x_dimension, y_dimension))

# #pdb.set_trace()
# fn_out= sample_transformer(temp_input, temp_input, temp_input, training=False, 
#                                enc_padding_mask=None, 
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)
# pdb.set_trace()
#fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)


#d_model = 128
dff = d_model


input_vocab_size = 8500# tokenizer_pt.vocab_size + 2
target_vocab_size = 8000# tokenizer_en.vocab_size + 2
#dropout_rate = 0.25

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()
#     #pdb.set_trace()
#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps
    
#   def __call__(self, step):
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)
#     #pdb.set_trace()
#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# learning_rate_custom = CustomSchedule(d_model)
# temp_learning_rate_schedule = CustomSchedule(d_model)

#plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
#plt.ylabel("Learning Rate")
#plt.xlabel("Train Step")
#plt.savefig("LR.png")

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# optimizer = tf.keras.optimizers.SGD(learning_rate)
# temp_learning_rate_schedule = CustomSchedule(d_model)

# #plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# #plt.ylabel("Learning Rate")
# #plt.xlabel("Train Step")
def my_loss(real, pred):
    #pdb.set_trace()
    #pred=tf.clip_by_value(pred, clip_value_min=1e-40, clip_value_max=1)
    #tf.print("Pred is: ", pred)
    #tf.print("Log pred is:", tf.math.log(pred))
    #tf.print("multy is: ", tf.math.multiply(real, tf.math.log(pred)))
    #tf.print("sum is: ", -tf.math.reduce_sum(tf.math.multiply(real, tf.math.log(pred)), 1))
    #manual_cost=tf.reduce_mean(-tf.math.reduce_sum(tf.math.multiply(real, tf.math.log(pred + 1e-10)), 1))
    loss=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real)  #+ L1 REGULARIZATION  
    #pred=tf.nn.softmax(pred)
    #manual_cost=tf.reduce_mean(-tf.math.reduce_sum(tf.math.multiply(real, tf.math.log(tf.nn.softmax(pred) + 1e-10)), 1) + (1-tf.math.abs(tf.nn.softmax(pred)[:,0]-tf.nn.softmax(pred)[:,1]))) 
    #tf.print("Manual loss is: ", manual_cost)
    #tf.reduce_mean(tf.math.reduce_sum(sc.xlog1py(real, pred), 1))
    #pdb.set_trace()
    #return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=real)
    #cost_f = tf.reduce_mean(sc.xlog1py(real, pred))# real * tf.log(pred))
    #cost_s =  sc.xlog1py((1 - real), (1 - pred)) # (1 - real) * tf.log(1 - pred)
    #cost=-tf.reduce_mean((real * tf.log(pred))  + (1 - real) * tf.log(1 - pred)    )
    #cost = - tf.reduce_mean( sc.xlog1py(real, pred)  +  sc.xlog1py((1 - real) , (1 - pred)))
    #manual_cost=tf.reduce_mean(-tf.reduce_sum(real * tf.log(pred), reduction_indices=[1]))
    return tf.reduce_mean(loss)#tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real)
    #cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    #return cce(y_true= real, y_pred=pred)

#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')

# def loss_function(real, pred):
#   #pdb.set_trace()
#   ##mask = tf.math.logical_not(tf.math.equal(real, 0))
#   print("===============================")
#   print("===============================")
#   print("===============================")
#   print(real)
#   print(pred)
#   print("===============================")
#   print("===============================")
#   print("===============================")  
#   loss_ = loss_object(real, pred)

#   ##mask = tf.cast(mask, dtype=loss_.dtype)
#   ##loss_ *= mask
  
#   return tf.reduce_mean(loss_)

#train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#     name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

# def create_masks(medications, tar):
#   # Encoder padding mask
#   enc_padding_mask = create_padding_mask(medications)
  
#   # Used in the 2nd attention block in the decoder.
#   # This padding mask is used to mask the encoder outputs.
#   dec_padding_mask = create_padding_mask(medications)
  
#   # Used in the 1st attention block in the decoder.
#   # It is used to pad and mask future tokens in the input received by 
#   # the decoder.
#   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#   dec_target_padding_mask = create_padding_mask(tar)
#   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
#   return enc_padding_mask, combined_mask, dec_padding_mask

# checkpoint_path = "./checkpoints/train"

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
#===========================SAVING MODELS==================
# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')
#==========================================================
# train_step_signature = [
#     tf.TensorSpec(shape=(BATCH_SIZE, num_time_steps, d_meds), dtype=np.float32),
#     tf.TensorSpec(shape=(BATCH_SIZE, num_time_steps, d_diags), dtype=np.float32),
#     tf.TensorSpec(shape=(BATCH_SIZE, num_classes), dtype=np.float32),
# ]

# @tf.function(input_signature=train_step_signature)
# def train_step(medications, diagnoses, demog_info, tar):
#   #pdb.set_trace()
#   #tar_inp = tar#[:, :-1]
#   #tar_real = tar#[:, 1:]
#   #enc_padding_mask=tf.to_float(tf.random.uniform((84,1,1,12), minval=0, maxval=1, dtype=tf.float32))


# ======= Restoring the model
# pdb.set_trace()
checkpoint_path =  "./checkpoints/trained_model_" +str(best_model_number) 
checkpoint = tf.train.Checkpoint(transformer=transformer)
# c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=optimum_epoch)#, ...)
c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=EPOCHS)#, ...)

if c_manager.latest_checkpoint:
    tf.print("-----------Restoring from {}-----------".format(
        c_manager.latest_checkpoint))
    checkpoint.restore(c_manager.latest_checkpoint)
    # optimum_epoch = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]

# if optimum_epoch == '':
#     if c_manager.latest_checkpoint:
#         tf.print("-----------Restoring from {}-----------".format(
#             c_manager.latest_checkpoint))
#         checkpoint.restore(c_manager.latest_checkpoint)
#         optimum_epoch = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]
#     else:
#         tf.print("-----------Initializing from scratch-----------")
# else:    
#     checkpoint_fname = checkpoint_path + '/'+ 'ckpt-' + str(optimum_epoch)
#     tf.print("-----------Restoring from {}-----------".format(checkpoint_fname))
#     checkpoint.restore(checkpoint_fname)


# model_path =  "./checkpoints/trained_model_" +str(best_model_number) + '_' + str(optimum_epoch) + '/'
# transformer.restoreVariables()
# checkpoint_path =  "./checkpoints/trained_model_" +str(best_model_number) #+ '_' + str(optimum_epoch)
# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)                           
# if checkpoint_path+'/ckpt-'+str(optimum_epoch):
#   ckpt.restore(checkpoint_path+'/ckpt-'+str(optimum_epoch)).assert_consumed()
#   #ckpt.restore(ckpt_manager.latest_checkpoint).assert_consumed()
# else:
#   pdb.set_trace()
#   print("Error: checkpoint wasn't found!")


# checkpoint_path =  "./checkpoints/trained_model_" +str(best_model_number) + '_' + str(optimum_epoch)
# ckpt = tf.train.Checkpoint(transformer=transformer)#, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint).assert_consumed()
#   print ('Latest checkpoint restored-TWO STREAM NODEL!!')
#   print('MOdel number is: ', best_model_number)
#   print('Early stopping epoch is, ', optimum_epoch)

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
#========================================
#pdb.set_trace()
# sess = tf.Session()
# variables_names = [v.name for v in transformer.trainable_variables()]
# values = sess.run(variables_names)
# for k, v in zip(variables_names, values):
#     print("Variable: ", k)
#     print("Shape: ", v.shape)
#     print(v)
# early_stopping=False


#=========================== Testing
#===== Breaking test set to k sets
test_split_k = 9803          
diagnoses_test_split=tf.split(value=diagnoses_test, num_or_size_splits=test_split_k)
medications_test_split=tf.split(value=medications_test, num_or_size_splits=test_split_k)
test_demog_info_split=tf.split(value=test_demog_info, num_or_size_splits=test_split_k)
logits_test_all = list()
# logits_test_all = np.zeros((len(medications_test), num_classes))
for i in range(len(diagnoses_test_split)):
  val_enc_mask = create_padding_mask(medications_test_split[i], diagnoses_test_split[i])
  logits_test = transformer(medications_test_split[i], diagnoses_test_split[i], test_demog_info_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
  logits_test_all.extend(logits_test)

# test_enc_mask = create_padding_mask(medications_test, diagnoses_test)
# logits_test = transformer(medications_test, diagnoses_test, test_demog_info, False, enc_padding_mask=test_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
# pdb.set_trace()
predictions_test_soft=tf.nn.softmax(logits_test_all)
np.savetxt("restoring_best_model_logits_test_twostreamT_"+str(model_number)+".csv", logits_test_all, delimiter=",")
np.savetxt("restoring_best_model__softmax_test_"+str(model_number)+".csv", predictions_test_soft.numpy(), delimiter=",")
# np.savetxt("logits_test_twostreamT_"+str(model_number)+".csv", logits_test.numpy(), delimiter=",")
# np.savetxt("softmax_test_"+str(model_number)+".csv", predictions_test_soft.numpy(), delimiter=",")
#predictions_test= tf.math.argmax(predictions_test_soft, 1) 
#predictions_test_posVect=[0 if x==1 else 1 for x in predictions_test.numpy()]    

#============ Thresholding for test=================
probabilities_test_pos=predictions_test_soft.numpy()[:,0]
# current_threshold_temp=0
# thresholding_results=np.zeros((num_thresholds, 11)) 
# y = np.array([1, 1, 2, 2])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# pdb.set_trace()
fpr, tpr, thresholds = metrics.roc_curve(testset_labels[:,0], probabilities_test_pos, pos_label=1)
test_auc = metrics.auc(fpr, tpr)
# rf_test_auc=roc_auc_score(testset_labels[:,0], best_rf_model.predict_proba(test_data_ar[:,1:-2])[:,1])
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
      # results_file.write(str(optimum_epoch))
      # results_file.write(",")  
      results_file.write(str(regu_factor))
      results_file.write("\n")  
