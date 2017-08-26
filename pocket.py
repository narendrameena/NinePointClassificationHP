import math
from random import choice
from numpy import array, dot, random
import pre_process_data_nine_point as pre


unit_step = lambda x: 0 if x < 0 else 1 

ff=open('pocket_results.csv','w')
ff.write('Iterations\tErrors\tTest set length\terror %\n')

for iteration in range(0,1000):
        

        pos_data=pre.pos_data
        pos_a=pre.pos_a
        neg_data=pre.neg_data
        neg_a=pre.neg_a

        percent=0.66

        train_len_pos=int(math.ceil(len(pos_data)*percent))
        train_len_neg=int(math.ceil(len(neg_data)*percent))
        test_len_pos=len(pos_data)-train_len_pos
        test_len_neg=len(neg_data)-train_len_neg

        training_data=[]
        for i in range(0,train_len_pos):
                training_data.append(tuple([array(pos_data[i]),1]))

        for i in range(0,train_len_neg):
                training_data.append(tuple([array(neg_data[i]),0]))

        test_data=[]
        for i in range(train_len_pos+1,len(pos_data)):
                test_data.append(tuple([array(pos_data[i]),1]))

        for i in range(train_len_neg+1,len(neg_data)):
                test_data.append(tuple([array(neg_data[i]),0]))



        #training_data = [(array([0, 0, 1]), 0), (array([0, 1, 1]), 1), (array([1, 0, 1]), 1), (array([1, 1, 1]), 1), ] 

        w = random.rand(9) 
        errors = [] 
        eta = 0.2 
        n = 1000
        count=0
        for i in xrange(n): 
                x, expected = choice(training_data) 
                result = dot(w, x) 
                error = expected - unit_step(result) 
                errors.append(error) 
                w += eta * error * x



        '''
        for x, _ in training_data: 
                result = dot(x, w)
                #print("{}: {} -> {}".format(x[: 2], result, unit_step(result)))
                print("{}: {} -> {}".format(x, result, unit_step(result)))
        '''

        #Testing
        for y, _ in test_data:
            final = dot(w, y)
            errors = unit_step(final) - _
            #print("{}: {} -> {}, {}, error : {}".format(y, final, unit_step(final), _, errors))
            
            if errors != 0:
                count += 1

        print count
        ff.write(str(iteration)+'\t'+str(count)+'\t'+str(len(test_data))+'\n')
ff.close()
    
