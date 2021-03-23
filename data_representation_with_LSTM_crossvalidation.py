"""
Created in March 2021

@author: Sajjad Fouladvand, Institute for Biomedical Informatics, University of Kentucky, Lexington KY ==> sjjd.fouladvand@gmail.com or fouladvand@uky.edu

This code train a LSTM model using the medication streams and represent the medication stream using the LSTM. 
This code is for parameter tuning and the LSTM model designing and structure is in data_representation_with_LSTM_model
"""

import  data_representation_with_LSTM_model as lsad
import os
import random as rnd
import pdb
import numpy as np
import random

main_counter=0
num_random_searches=10

train_meds_filename='YOUR_TRAINING_MEDICATION_STREAM.csv'
train_labels_filename='training_labels_shuffled_balanced.csv' # Labels in the same order as the order of samples in the training medication stream

validation_meds_filename='YOUR_VALIDATION_MEDICATION_STREAM.csv'
validation_labels_filename='validation_labels_shuffled_balanced.csv' # Labels in the same order as the order of samples in the validation medication stream
    
header_results_filename= "Learning Rate, Number of hidden neurons, Batch Size, accuracy, Precision, Recall, F1_score, specificity, TP, TN, FP,FN, Num Iterations, AUC, Regularization coefficient\n"
validation_results=np.zeros((num_random_searches, 15))
with open('Results_lstm_medications_validation_balanced.csv', 'w') as res_f:  
    res_f.write("".join(["".join(x) for x in header_results_filename]))      
    learning_rate_ar=np.random.uniform(0.000001,0.1,num_random_searches)
    num_hidden_array=np.random.randint(20,200,num_random_searches)
    batch_size_array=2** np.random.randint(5,9,num_random_searches)
    while (main_counter < num_random_searches):
        print("====================================") 
        print("Main counter is:")
        print("====================================")
        print(main_counter)
        learning_rt= random.choice([0.01, 0.001, 0.0001])
        n_hid=10 # Note, the number of hidden neurons is fixed to 10. The reason is that the medication stream dimension of the MUPOD is fixed to 10 
        batch_sz=random.choice([64, 256, 512])
        training_iters_up=random.choice([10000, 50000, 100000,200000])
        reg_coeff =random.choice([0.000001, 0.00001, 0.0001])
        j=0
        while(j<1): # This is to use when we need to average the validation results. But because it is time consuming we dont use it at this stage.
            j=j+1
            # Sending the current set of hyper-parameters to the LSTM model
            accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, validation_auc=lsad.main(main_counter, False, training_iters_up, reg_coeff, learning_rt,n_hid, batch_sz,train_meds_filename,train_labels_filename,validation_meds_filename,validation_labels_filename)
            # Writing down the performance of the current model.
            res_f.write(str(learning_rt))
            validation_results[main_counter, 0]=learning_rt
            res_f.write(", ")
            res_f.write(str(n_hid))
            validation_results[main_counter, 1]=n_hid
            res_f.write(", ")
            res_f.write(str(batch_sz))
            validation_results[main_counter, 2]=batch_sz
            res_f.write(", ")
            res_f.write(str(accuracy_temp))
            validation_results[main_counter, 3]=accuracy_temp
            res_f.write(", ")
            res_f.write(str(precision))
            validation_results[main_counter, 4]=precision
            res_f.write(", ")
            res_f.write(str(recall))
            validation_results[main_counter, 5]=recall
            res_f.write(", ")
            if(precision+recall ==0):
                F1=0
            else:    
                F1=(2*precision*recall)/(precision+recall)
            res_f.write(str(F1))
            validation_results[main_counter, 6]=F1
            res_f.write(", ")
            res_f.write(str(specificity))
            validation_results[main_counter, 7]=specificity
            res_f.write(", ")
            res_f.write(str(tp))
            validation_results[main_counter, 8]=tp
            res_f.write(", ")
            res_f.write(str(tn))
            validation_results[main_counter, 9]=tn
            res_f.write(", ")
            res_f.write(str(fp))
            validation_results[main_counter, 10]=fp
            res_f.write(", ")
            res_f.write(str(fn))
            validation_results[main_counter, 11]=fn
            res_f.write(", ")
            res_f.write(str(training_iters_up))
            validation_results[main_counter, 12]=training_iters_up
            res_f.write(",") 
            res_f.write(str(validation_auc))
            validation_results[main_counter, 13]=validation_auc
            res_f.write(",") 
            res_f.write(str(reg_coeff))
            validation_results[main_counter, 14] = reg_coeff
            res_f.write("\n")                                             
        res_f.write("\n")         
        main_counter=main_counter+1       
# Finding the best model based on their performances on validation set and then testing it on the test set AND representing the data.
max_f1_index=np.argmax(validation_results, axis=0)[6]

train_meds_filename='YOUR_TRAINING_MEDICATION_STREAM.csv'
train_labels_filename='training_labels_shuffled_balanced.csv'

test_meds_filename='YOUR_TESTING_MEDICATION_STREAM.csv'
test_labels_filename='testing_labels_shuffled_balanced.csv'

accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, test_auc =lsad.main(1000, True, validation_results[max_f1_index,12], validation_results[max_f1_index,14], validation_results[max_f1_index,0],int(validation_results[max_f1_index,1]), int(validation_results[max_f1_index,2]),train_meds_filename,train_labels_filename,test_meds_filename,test_labels_filename)
with open('Results_lstm_medications_test_balanced.csv', 'w') as res_f:
        res_f.write("".join(["".join(x) for x in header_results_filename]))      
        res_f.write(str(validation_results[max_f1_index,0]))
        res_f.write(", ")
        res_f.write(str(validation_results[max_f1_index,1]))
        res_f.write(", ")
        res_f.write(str(validation_results[max_f1_index,2]))
        res_f.write(", ")
        res_f.write(str(accuracy_temp))
        res_f.write(", ")
        res_f.write(str(precision))
        res_f.write(", ")
        res_f.write(str(recall))
        res_f.write(", ")
        if (precision + recall) == 0:
            F1=0
        else:    
            F1=(2*precision*recall)/(precision+recall)
        res_f.write(str(F1))
        res_f.write(", ")
        res_f.write(str(specificity))
        res_f.write(", ")
        res_f.write(str(tp))
        res_f.write(", ")
        res_f.write(str(tn))
        res_f.write(", ")
        res_f.write(str(fp))
        res_f.write(", ")
        res_f.write(str(fn))
        res_f.write(",")
        res_f.write(str(validation_results[max_f1_index, 12]))
        res_f.write(",")  
        res_f.write(str(test_auc))
        res_f.write(",")  
        res_f.write(str(validation_results[max_f1_index,14]))
        res_f.write("\n")                        
