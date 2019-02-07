# _*_ encoding: utf-8 _*_
'''
Created on 2019年1月17日

@author: ZHOUGANG880
'''

import numpy as np


from data_gen import data_generator, x_test, y_test
from model import Deeplabv3

import matplotlib.pyplot as plt


batch_size=10
gen = data_generator(x_test, y_test, batch_size=batch_size)
x, y, _ = next(gen)

print('load model...')
basemodel = Deeplabv3(input_shape=(28, 28, 3), classes=11, backbone='mobilenetv2')
basemodel.load_weights('checkpoints/DeepLabV3+-Weights-120.hdf5', by_name=True)
print('load model done.')

logits = basemodel.predict(x)
logits = np.argmax(logits, axis=-1)

for i in range(batch_size):
    img = x[i]
    logit = logits[i]
    label = np.max(logit)
    
    # 展示图片和预测的mask以及分类label
    plt.subplot(121)
    plt.title('Image',fontsize='large',fontweight='bold')
    plt.imshow(img)
    plt.subplot(122)
    plt.title('mask with label:'+str(label-1),fontsize='large',fontweight='bold')
    plt.imshow(logit)
    plt.show()






