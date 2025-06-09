import numpy as np
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
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,  # Number of samples
        n_features=20,   # Total features
        n_informative=15,  # Informative features
        n_classes=2,     # Binary classification
        random_state=42  # For reproducibility
    )

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Define the model creation function
    def create_model(params):
        """
        Creates a neural network model based on the given hyperparameters.

        Args:
            params: A tuple containing (learning_rate, units, dropout_rate, l2_reg, batch_size).

        Returns:
            model: The compiled Keras model.
            batch_size: The batch size as an integer.
        """
        learning_rate, units, dropout_rate, l2_reg, batch_size = params

        model = Sequential([
            Dense(int(units), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model, int(batch_size)

    # Define the objective function for Bayesian optimization
    def objective_function(hyperparams):
        """
        Objective function for Bayesian Optimization.

        Args:
            hyperparams: A list containing the hyperparameters to evaluate.

        Returns:
            Negative validation accuracy (to minimize).
        """
        # Extract hyperparameters
        learning_rate, units, dropout_rate, l2_reg, batch_size = hyperparams[0]

        # Create the model
        model, batch_size = create_model((learning_rate, units, dropout_rate, l2_reg, batch_size))

        # Define callbacks
        checkpoint_path = f"model_lr{learning_rate:.4f}_units{int(units)}_dropout{dropout_rate:.2f}_l2{l2_reg:.4f}_batch{int(batch_size)}.h5"
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=batch_size,
            verbose=0,
            callbacks=[checkpoint, early_stopping]
        )

        # Get the best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])

        return -val_accuracy  # Minimize negative accuracy

    # Define the hyperparameter space
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
    ]

    # Run Bayesian Optimization
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,  # Objective function
        domain=bounds,         # Hyperparameter space
        maximize=False         # Minimize the objective
    )

    optimizer.run_optimization(max_iter=30)  # Number of iterations

    # Print the best parameters and their corresponding accuracy
    print("Best parameters:", optimizer.x_opt)
    print("Best accuracy:", -optimizer.fx_opt)

    # Save optimization results
    with open('bayes_opt.txt', 'w') as f:
        f.write("Optimal hyperparameters:\n")
        f.write(f"Learning rate: {optimizer.x_opt[0]:.4f}\n")
        f.write(f"Units: {int(optimizer.x_opt[1])}\n")
        f.write(f"Dropout rate: {optimizer.x_opt[2]:.2f}\n")
        f.write(f"L2 regularization: {optimizer.x_opt[3]:.4f}\n")
        f.write(f"Batch size: {int(optimizer.x_opt[4])}\n")
        f.write(f"Best validation accuracy: {-optimizer.fx_opt:.4f}\n")

    # Plot convergence
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')
    plt.show()