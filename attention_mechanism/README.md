# Implementatino of attention mechanism in LSTMs for stock prediction

- Install necessary libraries listed in
  **requirements.txt** file using **pip install**
- Once installed, import those libraries and
  define the constants which are later used in the code and setup logging.
- I have written a utility function named
  **flatten_multiindex_columns()** whose objective is to flatten the columns with multiple indices.
- #Here is a sample dataframe with 5 columns:

      Column          Non-Null Count  Dtype

---  ------          --------------  -----
 0   (Close, AAPL)   63 non-null     float64
 1   (High, AAPL)    63 non-null     float64
 2   (Low, AAPL)     63 non-null     float64
 3   (Open, AAPL)    63 non-null     float64
 4   (Volume, AAPL)  63 non-null     int64

- I have bnow created a class named
  **"StockDataHandler"** that handles fetching, preprocessing and sequencing the stock data using the functions:

  **load_data()** - downloads the stock data from yahoo finance.
  **preprocess_data()** - preprocesses the data by scaling, sequencing and handling missing values.
  **scale_data()** - scales the data using MinMaxScaler().
  **create_sequences()** - creates sequences of data for LSTM input.**get_last_sequence()** - gets the last sequence for prediction. **get_original_data()** - returns the original data.
  **get_scaler()** - returns the scaler object.

- I have also written
  **LSTMAttentionModel** class that builds, trains, and uses the LSTM with Attention mechanism. Here are the functions in this class that deal with specific task:
  **build_model()** - builds the Keras Functional API model.
  **train()** - trains the model on the data downloaded from yahoo finance.
  **evaluate()** - evaluates the model on test data.
  **predict_future()** - predicts future stock prices iteratively for the next 4 days.

- **StockVisualizer** is another class
  that handles plotting of stock data and predictions.

- Finally, the main execution logic that
  calls all the necessary functions for building, training, testing the model and building the predictions.

# ([Simple implementation of the attention mechanism from scratch][def])

- how attention helped models like RNNs mitigate the vanishing gradient problem and capture long-range dependencies among words.
- Attention Mechanism is often associated with the transformer architecture, but it was already used in RNNs to mitigate Vanishing Gradient problem and capture more long-range dependencies mong words.

**Self-Attention in Transformers**:
classic attention indicates where words in the output sequence should focus attention in relation to the words in input sequence. Important in tasks like Machine Translation.

**Self-attention** operates between any two elements in the same sequence and provides information on how "correlated" the words are in the same sentence.

For a given token (or word) in a sequence, self-attention generates a list of attention weights corresponding to all other tokens in the sequence. This process is applied to each token in the sentence, obtaining a matrix of attention weights.

[def]: https://towardsdatascience.com/a-simple-implementation-of-the-attention-mechanism-from-scratch/
