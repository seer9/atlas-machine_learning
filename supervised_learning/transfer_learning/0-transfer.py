#!/usr/bin/env python3
""" model building with transfer learning """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ preprocesses the data"""
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    """load the base model"""
    base_model = K.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg',
        classes=10,
        classifier_activation='softmax'
    )

    """freeze the base model layers"""
    base_model.trainable = False

    """add custom layers"""
    new_model = K.Sequential()

    """layer for resizing the input images"""
    new_model.add(K.layers.Lambda(lambda x: K.backend.resize_images(x, 7, 7, "channels_last"), 
                                  input_shape=(32, 32, 3)))
    new_model.add(base_model)
    new_model.add(K.layers.Flatten())
    new_model.add(K.layers.Dense(512, activation='relu'))
    new_model.add(K.layers.Dropout(0.5))
    new_model.add(K.layers.Dense(10, activation='softmax'))

    """compile the model"""
    new_model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    """train the model"""
    new_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=64,
        epochs=10,
        verbose=1
    )

    """save the model"""
    model_path = '/home/clay/Documents/atlas-machine_learning/supervised_learning/transfer_learning/cifar10.h5'
    new_model.save(model_path)
    print(f"Model saved at {model_path}.")

