import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.models import Model
import cv2
import pickle as p


batch_size = 32
img_rows, img_cols = 208, 176
input_shape = (img_rows, img_cols, 3)


def load_data(seq):
    
    X_name = './SelectedData/X_train{:d}.data'.format(seq)
    Y_name = './SelectedData/Y_train{:d}.data'.format(seq)
    f = open(X_name,'rb')
    X = p.load(f)
    f = open(Y_name,'rb')
    Y = p.load(f)
    X = X.astype('float32')
    Y = Y.astype('float32')
    
    index = [i for i in range(X.shape[0])]
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    X = X.reshape(X.shape[0], img_rows, img_cols, 3)
    Y = Y.reshape(Y.shape[0], img_rows, img_cols, 1)
    
    return X, Y


#构建模型
inputs = Input(input_shape)

conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([conv4,up6], axis = 3)
conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv10)



#编译模型
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])


#评估模型
X_test, Y_test = load_data(11)


#训练模型
for times in range(50):
    for i in range(10):
        
        print('\ntimes: ', times+1, '/50\t', 'i: ', i+1, '/10\n')
        X_train, Y_train = load_data(i+1)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, Y_test))



model.save('./Model/modelN101.h5')

#评估模型
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


#结果输出
Output = model.predict(X_test, batch_size=batch_size)

Output[:,:,:][np.where(Output[:,:,:] < 0.5)] = 0
Output[:,:,:][np.where(Output[:,:,:] >= 0.5)] = 255
Y_test[:,:,:][np.where(Y_test[:,:,:] < 0.5)] = 0
Y_test[:,:,:][np.where(Y_test[:,:,:] >= 0.5)] = 255



for i in range(0, 1000):
    
    X_path = './NewOutput2/{:d}A.jpg'.format(i)
    Y_path = './NewOutput2/{:d}D.jpg'.format(i)
    Out_path = './NewOutput2/{:d}E.jpg'.format(i)
    
    cv2.imwrite(X_path, X_test[i, :, :, :])
    cv2.imwrite(Y_path, Y_test[i, :, :, :])
    cv2.imwrite(Out_path, Output[i, :, :, :])


