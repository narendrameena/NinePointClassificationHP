import numpy as np
#import redis, redisbayes

#rb = redisbayes.RedisBayes(redis=redis.Redis())


def preprocess(file):
    pos=np.loadtxt(file,dtype=type(str))
    pos=pos[1:]

    # a very complicated, 3 loop, one liner
    #to split the list of integers(currently as str),
    #then convert each one, in their respective rows, to int individually
    a=np.array([[int(j[k]) for k in range(0,len(j))] for j in (pos[i][0].split(',') for i in range(0,len(pos)))])
    data=a[:,range(1,a.shape[1]-1)]
    return(a,data)



pos_a,pos_data=preprocess('numpy_input_HP_pos.csv')
neg_a,neg_data=preprocess('numpy_input_HP_neg.csv')


'''
for i in range(0,len(pos_data)):
    rb.train('HP',pos_data[i])

for i in range(0,len(neg_data)):
    rb.train('HP',neg_data[i])

'''
