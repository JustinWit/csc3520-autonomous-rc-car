# img_filter_CNN.py
'''neural network to identify steering and throttle values based on a filtered image'''
from read_data import load_data
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pdb

def main():
    # load data 
    X, Y = load_data('data')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    y_throttle_train = y_train[:, 0]
    y_steering_train = y_train[:, 1]

    y_throttle_test = y_test[:, 0]
    y_steering_test = y_test[:, 1]

    pdb.set_trace()

    # create model
    img_in = (Input(shape=(120, 160, 3)))
    x = Conv2D(filters=24, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_1')(img_in)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_2')(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation='relu', name='conv2d_3')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', name='conv2d_4')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', name='conv2d_5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(.2)(x)
    x = Dense(100, activation='relu', name='dense_2')(x)
    x = Dropout(.2)(x)
    x = Dense(50, activation='relu', name='dense_3')(x)
    x = Dropout(.2)(x)

    output_1 = Dense(1, activation='linear', name='output_1')(x)

    model_steer = Model(inputs=[img_in], outputs=[output_1], name='gremlin')
    model_steer.compile(Adam(learning_rate=.001), loss='mse', metrics=['loss'])
    model_steer.summary()
    input("<Enter> to contiue")

    callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=5,
                          min_delta=0.001,
                          verbose = 1)]

    steering = model_steer.fit(
        x = X_train, 
        y = y_steering_train, 
        steps_per_epoch=50, 
        batch_size=100, 
        validation_data=(X_test, y_steering_test), 
        epochs=100, 
        verbose=1,
        callbacks=callbacks)

    model_throttle = Model(inputs=[img_in], outputs=[output_1], name='gremlin')
    model_throttle.compile(Adam(learning_rate=.001), loss='mse', metrics=['loss'])
    model_throttle.summary()

    

    throttle = model_throttle.fit(
        x = X_train, 
        y = y_throttle_train, 
        steps_per_epoch=50, 
        batch_size=100, 
        validation_data=(X_test, y_throttle_test), 
        epochs=100, 
        verbose=1,
        callbacks=callbacks)


    throttle_metrics = model_throttle.evaluate(X_test, y_throttle_test)
    steering_metrics = model_steer.evaluate(X_test, y_steering_test)
    print(f'Throttle loss: {throttle_metrics:0.2f}')
    print(f'Steering loss: {steering_metrics:0.2f}')
    print(f'Total loss: { throttle_metrics + steering_metrics:0.2f}')


if __name__ == "__main__":
    main()