# img_filter_CNN.py
'''neural network to identify steering and throttle values based on a filtered image'''
from read_data import load_data
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

    # load a saved model
    if args.loadmodels:
        model = load_model('models\\img_filter_CNN')
    
    # create a new model
    else:
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
        output_2 = Dense(1, activation='linear', name='output_2')(x)

        model = Model(inputs=[img_in], outputs=[output_1, output_2], name='gremlin')
        model.compile(Adam(learning_rate=.001), loss='mse')
        model.summary()
        # input("<Enter> to continue")

        callbacks = [
                EarlyStopping(monitor='val_loss',
                            patience=5,
                            min_delta=0.001,
                            verbose = 1)]

        model.fit(
            x = X_train, 
            y = y_train, 
            steps_per_epoch=50, 
            batch_size=100, 
            validation_data=(X_test, y_test), 
            epochs=100, 
            verbose=1,
            callbacks=callbacks)

        model.save('models\\img_filter_CNN')


    metrics = model.evaluate(X_test, y_test)
    print(f'Throttle loss: {metrics[1]:0.4}')
    print(f'Steering loss: {metrics[2]:0.4}')
    print(f'Total loss: {metrics[0]:0.4}')

if __name__ == "__main__":
    main(parser.parse_args())