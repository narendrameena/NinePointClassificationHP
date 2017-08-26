import math
from random import choice, sample
#import random 
from numpy import array, dot, random
import pre_process_data_nine_point as pre

from sklearn.naive_bayes import GaussianNB
import numpy as np



ff=open('naivebayes_results.csv','w')
ff.write('Iterations\tErrors\tTest set length\terror %\n')

for iteration in range(0,1000):

        pos_data=pre.pos_data
        pos_a=pre.pos_a
        neg_data=pre.neg_data
        neg_a=pre.neg_a

        percent=0.80

        train_len_pos=int(math.ceil(len(pos_data)*percent))
        train_len_neg=int(math.ceil(len(neg_data)*percent))
        test_len_pos=len(pos_data)-train_len_pos
        test_len_neg=len(neg_data)-train_len_neg

        train_sample_pos=sample(range(0,len(pos_data)),train_len_pos)
        train_sample_neg=sample(range(0,len(neg_data)),train_len_neg)
        test_sample_pos=list(set(range(0,len(pos_data)))-set(train_sample_pos))
        test_sample_neg=list(set(range(0,len(neg_data)))-set(train_sample_neg))


        X=[]
        y=[]
        '''
        for i in range(0,train_len_pos):
                X.append(pos_data[i])
                y.append(1)

        for i in range(0,train_len_neg):
                X.append(neg_data[i])
                y.append(0)
        '''
        for i in train_sample_pos:
                X.append(pos_data[i])
                y.append(1)

        for i in train_sample_neg:
                X.append(neg_data[i])
                y.append(0)


        test_X=[]
        test_y=[]
        '''
        for i in range(train_len_pos+1,len(pos_data)):
                test_X.append(pos_data[i])
                test_y.append(1)

        for i in range(train_len_neg+1,len(neg_data)):
                test_X.append(neg_data[i])
                test_y.append(0)
        '''
        for i in test_sample_pos:
                test_X.append(pos_data[i])
                test_y.append(1)

        for i in test_sample_neg:
                test_X.append(neg_data[i])
                test_y.append(0)




        #assigning predictor and target variables
        #x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
        #y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

        #Create a Gaussian Classifier
        model = GaussianNB()

        # Train the model using the training sets 
        model.fit(X, y)


        #testing
        #Predict Output 
        #predicted= model.predict([[1,2],[3,4]])
        #print predicted



        count=0
        for i in range(0,len(test_X)):
                ac=model.predict(test_X[i])
                error = 0 if ac[0]==test_y[i] else 1
                if error:
                        count+=1
                #print("{}: {} -> {}, error : {}".format(test_X[i], test_y[i], ac, error))

        print count

        #print "Given Split: "+str(percent*100)+"%\n"
        #print "Length of Training set: "+str(len(X))+" Length of Test set: "+str(len(test_X))+"\n"
        #print "Number of errors: "+str(count)
        #print "Accuracy: "+str((1-(float(count)/len(test_X)))*100)+"%"
        ff.write(str(iteration)+'\t'+str(count)+'\t'+str(len(test_X))+'\n')

ff.close()

