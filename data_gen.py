# _*_ encoding: utf-8 _*_
'''
Created on 2019年1月16日

@author: ZHOUGANG880
'''

import numpy as np


from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

def data_generator(x_data, y_data, batch_size=32):
    
    simple_num = len(x_data)
    step_per_epoch = simple_num // batch_size
    
    while True:
        images, labels = zip_shuffle(x_data, y_data)
        
        for i in range(step_per_epoch):
            x = images[i*batch_size:i*batch_size+batch_size]
            l = labels[i*batch_size:i*batch_size+batch_size]
            y = np.copy(x)
            
            for j in range(len(x)):
                p = np.where(x[j]!=0)
                y[j][p] = l[j]+1
            
            x, y = zip_expend_dim(x, y)
            yield x, y, l


def zip_expend_dim(x, y):
    bs,h,w = x.shape
    x = np.repeat(x, repeats=3)
    x.shape = (bs,h,w,3)
    y = np.expand_dims(y, axis=-1)
    return x, y


def zip_shuffle(x, y):
    p = np.random.permutation(range(len(x)))
    return x[p], y[p]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    gen = data_generator(x_train, y_train, batch_size=4)
    x, y, l = next(gen)
    
    for i in range(len(x)):
        img = x[i]
        img.shape=(28,28,3)
        mask = y[i]
        mask.shape=(28,28)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask,cmap='gray')
        plt.show()
    
    
    
    

