#!/usr/bin/env python3
"""tensor flow for NN creation"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
"""data gen and preprocessing"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""bayesian optimization"""
import GPyOpt
"""visualization"""
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # synthetic dataset for classifacation
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,  
        random_state=42  
    )

    # training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # the model creation function
    def create_model(params):

        learning_rate, units, dropout_rate, l2_reg, batch_size = params

        model = Sequential([
            Dense(int(units), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid') 
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model, int(batch_size)

    # define the objective function for Bayesian optimization
    def fitting(hyperparams):

        # initialize hyperparameters
        learning_rate, units, dropout_rate, l2_reg, batch_size = hyperparams[0]

        # initialize the model
        model, batch_size = create_model((learning_rate, units, dropout_rate, l2_reg, batch_size))

        # callbacks
        checkpoint_path = f"model_lr{learning_rate:.4f}_units{int(units)}_dropout{dropout_rate:.2f}_l2{l2_reg:.4f}_batch{int(batch_size)}.h5"
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=batch_size,
            verbose=0,
            callbacks=[checkpoint, early_stopping]
        )

        # get the best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])

        return -val_accuracy  # minimize negative accuracy

    # define the hyperparameter space
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
    ]

    # Bayesian Optimization
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=fitting,  
        domain=bounds,
        maximize=False
    )

    optimizer.run_optimization(max_iter=30)  # iterations

    # print the best parameters and their accuracy
    print("Best parameters:", optimizer.x_opt)
    print("Best accuracy:", -optimizer.fx_opt)

    # save results
    with open('bayes_opt.txt', 'w') as f:
        f.write("Optimal hyperparameters:\n")
        f.write(f"Learning rate: {optimizer.x_opt[0]:.4f}\n")
        f.write(f"Units: {int(optimizer.x_opt[1])}\n")
        f.write(f"Dropout rate: {optimizer.x_opt[2]:.2f}\n")
        f.write(f"L2 regularization: {optimizer.x_opt[3]:.4f}\n")
        f.write(f"Batch size: {int(optimizer.x_opt[4])}\n")
        f.write(f"Best validation accuracy: {-optimizer.fx_opt:.4f}\n")

    # plot convergence
    optimizer.plot_convergence()
    plt.show()