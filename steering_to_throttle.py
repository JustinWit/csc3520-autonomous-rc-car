# img_filter_CNN.py
'''neural network to identify steering and throttle values based on a filtered image'''
from read_data import load_data
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten, concatenate
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loadmodels',
                    help='boolean var if loading saved models',
                    action='store_true', 
                    default=False)

def main(args):
    # load data 
    X, Y = load_data('data')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # split labels by steer and throttle
    y_throttle_train = y_train[:, 0]
    y_steering_train = y_train[:, 1]

    y_throttle_test = y_test[:, 0]
    y_steering_test = y_test[:, 1]

    # load trained model from file
    if args.loadmodels:
        model_steering = load_model('models\\steering_to_throttle\\steering')
        model_throttle = load_model('models\\steering_to_throttle\\throttle')

        steering_out_train = model_steering.predict(X_train)
        steering_out_test = model_steering.predict(X_test)

    # train a new model
    else:
        # create model for steering
        img_in1 = (Input(shape=(120, 160, 3)))
        x1 = Conv2D(filters=24, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_1')(img_in1)
        x1 = Conv2D(filters=32, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_2')(x1)
        x1 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_3')(x1)
        x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', name='conv2d_4')(x1)
        x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', name='conv2d_5')(x1)
        x1 = Flatten(name='flatten')(x1)
        x1 = Dense(100, activation='relu', name='dense_1')(x1)
        x1 = Dropout(.2)(x1)
        x1 = Dense(100, activation='relu', name='dense_2')(x1)
        x1 = Dropout(.2)(x1)
        x1 = Dense(50, activation='relu', name='dense_3')(x1)
        x1 = Dropout(.2)(x1)

        output1 = Dense(1, activation='linear', name='output_1')(x1)

        model_steering = Model(inputs=[img_in1], outputs=[output1], name='steering_only')
        model_steering.compile(Adam(learning_rate=.001), loss='mse')
        model_steering.summary()
        # input("<Enter> to continue")

        callbacks = [
                EarlyStopping(monitor='val_loss',
                            patience=5,
                            min_delta=0.001,
                            verbose = 1)]

        steering = model_steering.fit(
            x = X_train, 
            y = y_steering_train, 
            steps_per_epoch=50, 
            batch_size=100, 
            validation_data=(X_test, y_steering_test), 
            epochs=100, 
            verbose=1,
            callbacks=callbacks)

        model_steering.save('models\\steering_to_throttle\\steering')

        # predict steering values to use as inputs to next network
        steering_out_train = model_steering.predict(X_train)
        steering_out_test = model_steering.predict(X_test)

        # throttle model# create model for throttle
        img_in2 = (Input(shape=(120, 160, 3)))
        x2 = Conv2D(filters=24, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_1')(img_in2)
        x2 = Conv2D(filters=32, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_2')(x2)
        x2 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_3')(x2)
        x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', name='conv2d_4')(x2)
        x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', name='conv2d_5')(x2)
        x2 = Flatten(name='flatten')(x2)
        x2 = Dense(100, activation='relu', name='dense_1')(x2)
        x2 = Dropout(.2)(x2)
        x2 = Dense(100, activation='relu', name='dense_2')(x2)
        x2 = Dropout(.2)(x2)
        x2 = Dense(50, activation='relu', name='dense_3')(x2)
        x2 = Dropout(.2)(x2)
        x2 = Dense(1, activation='linear', name='dense_4')(x2)
        
        # add input for steering values
        steer_in = Input(shape=(1, ))
        z = concatenate([x2, steer_in])

        output2 = Dense(1, activation='linear', name='output_1')(z)

        model_throttle = Model(inputs=[img_in2, steer_in], outputs=[output2], name='gremlin')
        model_throttle.compile(Adam(learning_rate=.001), loss='mse')
        model_throttle.summary()
        # input("<Enter> to continue")

        callbacks = [
                    EarlyStopping(monitor='val_loss',
                                patience=5,
                                min_delta=0.001,
                                verbose = 1)]

        throttle = model_throttle.fit(
            x = [X_train, steering_out_train], 
            y = y_throttle_train, 
            steps_per_epoch=50, 
            batch_size=100, 
            validation_data=([X_test, steering_out_test], y_throttle_test), 
            epochs=100, 
            verbose=1,
            callbacks=callbacks)

        model_throttle.save('models\\steering_to_throttle\\throttle')

    # results
    metrics_steering = model_steering.evaluate(X_test, y_steering_test)
    metrics_throttle = model_throttle.evaluate([X_test, steering_out_test], y_throttle_test)
    print(f'Steering loss: {metrics_steering:0.4f}')
    print(f'Throttle loss: {metrics_throttle:0.4f}')
    print(f'Total loss: {metrics_steering + metrics_throttle:0.4f}')
    


if __name__ == "__main__":
    main(parser.parse_args())