# _*_ encoding: utf-8 _*_
'''
Created on 2019年1月16日

@author: ZHOUGANG880
'''

import os

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import SGD

from keras.utils import multi_gpu_model

from model import Deeplabv3
from data_gen import x_train,y_train,x_test,y_test, data_generator


class ParallerModelCheckPoint(Callback):
    def __init__(self, single_model):
        self.mode_to_save = single_model
        
    def on_epoch_end(self, epoch, logs={}):
        print(r'save model: checkpoints/DeepLabV3+-Weights-%02d.hdf5'%(epoch+1))
        self.mode_to_save.save_weights(r'checkpoints/DeepLabV3+-Weights-%02d.hdf5'%(epoch+1))


# 设置使用的显存以及GPU
# 设置可用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# keras设置GPU参数
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.888
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
'''
use_gpu=False
gpus=2
batch_size=32
learning_rate=0.0001

# xception, mobilenetv2
basemodel = Deeplabv3(input_shape=(28, 28, 3), classes=11, backbone='mobilenetv2')

model_file = 'checkpoints/DeepLabV3+-Weights-40.hdf5'
if os.path.exists(model_file):
    print('loading model:', model_file)
    basemodel.load_weights(model_file, by_name=True)

if use_gpu:
    parallermodel = multi_gpu_model(basemodel, gpus=gpus)
    checkpoint = ParallerModelCheckPoint(basemodel)
else:
    parallermodel = basemodel
    checkpoint = ModelCheckpoint(r'checkpoints/DeepLabV3+-Weights-{epoch:02d}.hdf5', save_weights_only=True, verbose=1)
    
optimizer = SGD(lr=learning_rate, momentum=0.9, clipnorm=5.0)
parallermodel.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

train_gen = data_generator(x_train, y_train, batch_size=batch_size)
test_gen = data_generator(x_test, y_test, batch_size=batch_size)

tensorboard = TensorBoard('tflog', write_graph=True)

earlystop = EarlyStopping(monitor='acc', patience=10, verbose=1)

print('begin training...')
parallermodel.fit_generator(train_gen, 
                        steps_per_epoch=len(x_train)//batch_size, 
                        epochs=120, 
                        verbose=1, 
                        callbacks=[earlystop, checkpoint, tensorboard], 
                        validation_data=test_gen, 
                        validation_steps=100, 
                        initial_epoch=0)





