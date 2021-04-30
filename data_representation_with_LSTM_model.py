"""

Created in March 2021

@author: Sajjad Fouladvand, Institute for Biomedical Informatics, University of Kentucky, Lexington KY ==> sjjd.fouladvand@gmail.com or fouladvand@uky.edu

This code train a LSTM model using the medication streams and represent the medication stream using the LSTM. 

@title Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


"""

from __future__ import print_function

import os
import tensorflow as tf     
import random
import numpy as np
import csv
import pdb
from sklearn import metrics


# This function reads the medication stream and generates batches
class ReadingData(object):
    def __init__(self, path_t="", path_l="")::        
        self.data = []
        self.labels = []
        self.seqlen = []
        s=[]
        temp=[]
        with open(path_t) as f:
              data_header = next(f)  
              for line in f:
                  d_temp=line.split(',')
                  d_temp=[float(x) for x in d_temp]
                  self.data.append(d_temp)
                  d_temp=[]
                  s=[]
        d_temp=[]
        with open(path_l) as f:
              for line in f:
                  d_temp=[]
                  d_temp=line.split(',')
                  d_temp=[float(x) for x in d_temp]
                  self.labels.append(d_temp)                  
                  d_temp=[]
        self.batch_id = 0
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        if len(batch_data) < batch_size:
            batch_data = batch_data + (self.data[0:(batch_size - len(batch_data))])
            batch_labels = batch_labels + (self.labels[0:(batch_size - len(batch_labels))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels

def dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell)#,input_keep_prob=0.5, output_keep_prob=keep_prob)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    lstm_input = x
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    outputs_original=outputs
    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen-1)
    # Indexing
    output_before_idx =outputs
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    output_after_idx =outputs
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out'], states, outputs, outputs_original, output_before_idx, output_after_idx, lstm_input

# This function finds the lengths of the input streams
def find_length(sequence):
    length=0
    for i in range(len(sequence)):
        if sum(sequence[i]) != 0:
            length=i
    return (length+1)

def find_beg_end(sequence):
    beginning=-1
    end=-1
    for i in range(len(sequence)):
        if sum(sequence[i]) != 0:
            end=i
        if sum(sequence[i]) != 0 and end ==-1:
            beginning= i    
    return beginning+1, end+1    
def main(idx, representing, training_iters_up, reg_coeff, learning_rt,n_hid, batch_sz,train_meds_filename, train_labels_filename,validation_meds_filename,validation_labels_filename):
    tf.reset_default_graph() 
    print("Creating and training the LSTM model ...")
    learning_rate = learning_rt
    training_iters_up = training_iters_up
    training_iters_low = 10000
    batch_size = batch_sz
    display_step = 10
    loss_threshold = 0.0001
    n_hidden = n_hid # hidden layer num of features
    num_classes = 2 
    num_time_steps=120
    seq_max_len = 120
    d_meds=10
    one=1
    zero=0
    accuracies=[]
    # Reading training data
    trainset_meds = ReadingData(path_t=train_meds_filename, path_l=train_labels_filename)#, path_s=train_lengths_filename)
    train_set_shape = np.shape(trainset_meds.data)
    real_labels=np.zeros((batch_size, num_classes), dtype=np.float32)

    #========= Reading and preparing validation data
    validationset_meds = ReadingData(path_t=validation_meds_filename, path_l=validation_labels_filename)#,path_s=validation_lengths_filename)
    validation_data = validationset_meds.data
    validation_label = validationset_meds.labels
    validation_data_ar=np.array(validation_data)
    validation_medications=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_meds+1))  # +1 for visit date 
    validationset_labels=np.array(validation_label)

    #========== Findinmg the sequence lengths
    validation_seq_length=[]
    for i in range(len(validation_medications)):
        validation_seq_length.append(find_length(validation_medications[i][:,1:]))
    labels_validation = validationset_labels[:,one:].astype(np.float32)

# tf Graph input
    x = tf.placeholder("float", [None, num_time_steps, d_meds ])  # input sequence
    y = tf.placeholder("float", [None, num_classes])       # labels

# A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])               # sequence length

# Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input = dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden)

# Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) 
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    pred_arg=tf.argmax(pred,1)
    y_arg=tf.argmax(y,1)
    softmax_predictions = tf.nn.softmax(pred)

# Initializing the variables
    init = tf.global_variables_initializer()
   
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        step = 1            
        with open ('Loss_train_medications_'+str(idx)+'.csv', 'w') as loss_file , open('Loss_validation_medications_'+str(idx)+'.csv', 'w') as loss_file_val:
            while step * batch_size < training_iters_up:
                # Reading a batch of training data and preparing it for the LSTM model
                batch_x, batch_y = trainset_meds.next(batch_size)
                batch_y_ar=np.array(batch_y)
                real_labels=batch_y_ar[:,one:].astype(np.float32)
                batch_x_ar=np.array(batch_x)
                medications=np.reshape(batch_x_ar[:,one:],(batch_size, num_time_steps, d_meds+1))   
                current_batch_length=[]
                for i in range(len(medications)):
                    current_batch_length.append(find_length(medications[i][:,1:]))

                sess.run(optimizer, feed_dict={x: medications[:,:,1:], y: real_labels,
                                        seqlen: current_batch_length})  
                loss = sess.run(cost, feed_dict={x: medications[:,:,1:], y: real_labels, seqlen: current_batch_length})
                loss_file.write(str(loss))
                loss_file.write("\n")
                    
                #====== Validation loss
                val_loss=sess.run(cost, feed_dict={x: validation_medications[:,:,1:], y: labels_validation,seqlen: validation_seq_length})                    
                loss_file_val.write(str(val_loss))
                loss_file_val.write("\n")    
                if loss <= loss_threshold and step * batch_size > training_iters_low:
                    break
                step += 1   
                if step%10000 ==0:
                    print(step)
        saver.save(sess, save_path='saved_models_lstm_meds/LSTM_medications_model_'+str(int(idx))+'.ckpt')
        print("Optimization Finished!")

        [y_arg_temp, softmax_predictions_temp, pred_arg_temp] =sess.run([y_arg, softmax_predictions, pred_arg], feed_dict={x: validation_medications[:,:,1:], y: labels_validation,seqlen: validation_seq_length})
        y_arg_temp_forAUC = np.abs(y_arg_temp-1)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_arg_temp_forAUC, y_score=softmax_predictions_temp[:,0], pos_label=1) #== It's arg and so 1 means negative and 0 means positive
        validation_auc=metrics.auc(fpr, tpr)         
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(validation_medications)):
            if(pred_arg_temp[i]==1 and y_arg_temp[i]==1):
                tn=tn+1
            elif(pred_arg_temp[i]==0 and y_arg_temp[i]==0):
                tp=tp+1
            elif(pred_arg_temp[i]==0 and y_arg_temp[i]==1):
                fp=fp+1
            elif(pred_arg_temp[i]==1 and y_arg_temp[i]==0):
                fn=fn+1          
        if( (tp+fp)==0):
            precision=0
        else:
            precision=tp/(tp+fp)
        if (tp+fn) == 0:
            recall = 0
        else:    
            recall=tp/(tp+fn)
        if (tp+fn)==0:
            sensitivity=0
        else:    
            sensitivity=tp/(tp+fn)
        if (tn+fp)==0:
            specificity=0
        else:
            specificity=tn/(tn+fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        if representing == True:
            #============================== Represent patients using trained LSTM        
            print("Representing the data...")
            print("=================================")        
            with open("training_medications_represented.csv", 'w') as train_rep_file, open("validation_medications_represented.csv",'w') as valid_rep_file, open("testing_medications_represented.csv", 'w') as test_rep_file:
                train_meds_filename='YOUR_TRAINING_MEDICATION_STREAM.csv'
                train_labels_filename='training_labels_shuffled_balanced.csv'

                validation_meds_filename='YOUR_VALIDATION_MEDICATION_STREAM.csv'
                validation_labels_filename='validation_labels_shuffled_balanced.csv'            

                test_meds_filename='YOUR_TESTING_MEDICATION_STREAM.csv'
                test_labels_filename='testing_labels_shuffled_balanced.csv'
                
                #======================== Reading validation data
                validationset_meds = ReadingData(path_t=validation_meds_filename, path_l=validation_labels_filename)#,path_s=validation_lengths_filename)
                validation_data = validationset_meds.data
                validation_label = validationset_meds.labels
                validation_data_ar=np.array(validation_data)
                validation_medications=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_meds+1))   
                validationset_labels=np.array(validation_label)

                validation_seq_beg=[]
                validation_seq_end=[]
                for i in range(len(validation_medications)):
                    beginning, end = find_beg_end(validation_medications[i][:,1:])
                    validation_seq_beg.append(beginning)
                    validation_seq_end.append(end)
                labels_validation = validationset_labels[:,one:].astype(np.float32)
                #======================== Reading training data    
                training_meds = ReadingData(path_t=train_meds_filename, path_l=train_labels_filename)
                training_data = training_meds.data
                training_label = training_meds.labels
                training_data_ar=np.array(training_data)
                training_medications=np.reshape(training_data_ar[:,one:],(len(training_data_ar), num_time_steps, d_meds+1))   
                training_labels=np.array(training_label)


                training_seq_beg=[]
                training_seq_end=[]
                for i in range(len(training_medications)):
                    beginning, end = find_beg_end(training_medications[i][:,1:])
                    training_seq_beg.append(beginning)
                    training_seq_end.append(end)
                labels_training = training_labels[:,one:].astype(np.float32)    
                #======================== Reading testing data    
                testing_meds = ReadingData(path_t=test_meds_filename, path_l=test_labels_filename)
                testing_data = testing_meds.data
                testing_label = testing_meds.labels
                testing_data_ar=np.array(testing_data)
                testing_medications=np.reshape(testing_data_ar[:,one:],(len(testing_data_ar), num_time_steps, d_meds+1))   
                testing_labels=np.array(testing_label)

                testing_seq_beg=[]
                testing_seq_end=[]
                for i in range(len(testing_medications)):
                    beginning, end = find_beg_end(testing_medications[i][:,1:])
                    testing_seq_beg.append(beginning)
                    testing_seq_end.append(end)
                labels_testing = testing_labels[:,one:].astype(np.float32)        
                
                # #==================== Sampling data=====================
                # sample_size = 1000
                # testing_medications = testing_medications[:sample_size,:,:]
                # testing_seq_end = testing_seq_end[:sample_size]
                # testing_seq_beg = testing_seq_beg[:sample_size]
                # labels_testing = labels_testing[:sample_size,:]

                # validation_medications = validation_medications[:sample_size,:,:]
                # validation_seq_end = validation_seq_end[:sample_size]
                # validation_seq_beg = validation_seq_beg[:sample_size]
                # labels_validation = labels_validation[:sample_size,:]

                # training_medications = training_medications[:sample_size,:,:]
                # training_seq_end = training_seq_end[:sample_size]
                # training_seq_beg = training_seq_beg[:sample_size]
                # labels_training = labels_training[:sample_size,:]                

                # #=======================================================

                #================= Representing data        
                [states_val, outputs_original_val] =sess.run([states, outputs_original], feed_dict={x: validation_medications[:,:,1:], y: labels_validation,seqlen: validation_seq_end})            
                array_temp=np.array(outputs_original_val).flatten()
                array_temp_reshaped=np.reshape(array_temp,(num_time_steps, -1))
                start_idx=0
                labels_idx=0              
                while start_idx < array_temp_reshaped.shape[1]:
                    slice_temp = array_temp_reshaped[:,start_idx:start_idx+n_hidden]
                    current_patient = slice_temp.flatten()
                    current_patient_enrolid = validationset_labels[labels_idx,0]
                    valid_rep_file.write(str(current_patient_enrolid))
                    valid_rep_file.write(',')
                    valid_rep_file.write(','.join(map(repr, current_patient)))
                    valid_rep_file.write(',')
                    valid_rep_file.write(','.join(map(repr, validationset_labels[labels_idx, one:])))
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(float(validation_seq_beg[labels_idx])))
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(float(validation_seq_end[labels_idx])))                    
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(current_patient_enrolid))
                    valid_rep_file.write('\n')
                    start_idx += n_hidden 
                    labels_idx +=1   
                [states_train, outputs_original_train] =sess.run([states, outputs_original], feed_dict={x: training_medications[:,:,1:], y: labels_training,seqlen: training_seq_end})                     
                array_temp=np.array(outputs_original_train).flatten()
                array_temp_reshaped=np.reshape(array_temp,(num_time_steps, -1))
                start_idx=0
                labels_idx=0     
                while start_idx < array_temp_reshaped.shape[1]:
                    slice_temp = array_temp_reshaped[:,start_idx:start_idx+n_hidden]
                    current_patient = slice_temp.flatten()
                    current_patient_enrolid = training_labels[labels_idx,0]
                    train_rep_file.write(str(current_patient_enrolid))
                    train_rep_file.write(',')
                    train_rep_file.write(','.join(map(repr, current_patient)))
                    train_rep_file.write(',')
                    train_rep_file.write(','.join(map(repr, training_labels[labels_idx, one:])))
                    train_rep_file.write(',')
                    train_rep_file.write(str(float(training_seq_beg[labels_idx])))
                    train_rep_file.write(',')
                    train_rep_file.write(str(float(training_seq_end[labels_idx])))                    
                    train_rep_file.write(',')
                    train_rep_file.write(str(current_patient_enrolid))
                    train_rep_file.write('\n')
                    start_idx += n_hidden 
                    labels_idx +=1                        
                [states_test, outputs_original_test] =sess.run([states, outputs_original], feed_dict={x: testing_medications[:,:,1:], y: labels_testing,seqlen: testing_seq_end})                       
                array_temp=np.array(outputs_original_test).flatten()
                array_temp_reshaped=np.reshape(array_temp,(num_time_steps, -1))
                start_idx=0
                labels_idx=0                              
                while start_idx < array_temp_reshaped.shape[1]:
                    slice_temp = array_temp_reshaped[:,start_idx:start_idx+n_hidden]
                    current_patient = slice_temp.flatten()
                    current_patient_enrolid = testing_labels[labels_idx,0]
                    test_rep_file.write(str(current_patient_enrolid))
                    test_rep_file.write(',')
                    test_rep_file.write(','.join(map(repr, current_patient)))
                    test_rep_file.write(',')
                    test_rep_file.write(','.join(map(repr, testing_labels[labels_idx, one:])))
                    test_rep_file.write(',')
                    test_rep_file.write(str(float(testing_seq_beg[labels_idx])))
                    test_rep_file.write(',')
                    test_rep_file.write(str(float(testing_seq_end[labels_idx])))                    
                    test_rep_file.write(',')
                    test_rep_file.write(str(current_patient_enrolid))
                    test_rep_file.write('\n')
                    start_idx += n_hidden 
                    labels_idx +=1           
    return accuracy, precision, recall, sensitivity, specificity, tp, tn, fp, fn, validation_auc#, accuracy_temp_train, precision_train, recall_train, sensitivity_train, specificity_train,tp_train, tn_train, fp_train, fn_train

if __name__ == "__main__": main(idx,representing, training_iters_up, reg_coeff, learning_rt,n_hid, batch_sz,train_meds_filename,train_labels_filename,validation_meds_filename,validation_labels_filename)