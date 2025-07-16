# Project: Forecasting Bitcoin (BTC) Value Using RNNs  

## Objectives  
1. **Preprocess Data**: Create a script (`preprocess_data.py`) to clean and prepare the raw datasets (`coinbase` and `bitstamp`) for training.  
2. **Build and Train Model**: Develop a script (`forecast_btc.py`) to create, train, and validate a Keras RNN model for BTC forecasting.  

---

## Approach  

### Preprocessing (`preprocess_data.py`)  
1. **Load Data**:  
    - Read the `coinbase` and `bitstamp` datasets.  
    - Combine datasets if necessary.  

2. **Filter Data**:  
    - Remove irrelevant features (e.g., Unix time).  
    - Focus on features like open price, high price, low price, close price, and volume-weighted average price.  

3. **Rescale Data**:  
    - Normalize numerical features using Min-Max scaling or Standardization.  

4. **Aggregate Time Windows**:  
    - Group data into 24-hour windows to create input sequences.  
    - Use the close price of the next hour as the target variable.  

5. **Save Preprocessed Data**:  
    - Save the processed data in a format suitable for `tf.data.Dataset` (e.g., `.csv` or `.tfrecord`).  

---

### Model Development (`forecast_btc.py`)  
1. **Load Preprocessed Data**:  
    - Use `tf.data.Dataset` to load and batch the preprocessed data.  

2. **Define RNN Architecture**:  
    - Input: 24-hour sequence of BTC data.  
    - Layers:  
      - LSTM or GRU layers for temporal modeling.  
      - Dense layers for prediction.  
    - Output: Predicted close price for the next hour.  

3. **Compile Model**:  
    - Loss: Mean Squared Error (MSE).  
    - Optimizer: Adam or RMSprop.  

4. **Train Model**:  
    - Split data into training, validation, and test sets.  
    - Train the model using the training set.  
    - Validate the model using the validation set.  

5. **Evaluate Model**:  
    - Test the model on unseen data.  
    - Report metrics like MSE and visualization of predictions vs. actual values.  

---

## Deliverables  
1. `preprocess_data.py`: Script for data preprocessing.  
2. `forecast_btc.py`: Script for model creation, training, and validation.  
3. Preprocessed datasets saved in a reusable format.  
4. Trained RNN model and evaluation results.  

---  
## Notes  
- Ensure reproducibility by setting random seeds.  
- Consider hyperparameter tuning for optimal performance.  
- Use visualization tools (e.g., Matplotlib) to analyze predictions.  
