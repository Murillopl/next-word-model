# Next Word Model

🔮 **Next word prediction using an LSTM model trained on Shakespeare's Hamlet**

This project demonstrates how an LSTM neural network can be trained to predict the next word in a sequence of text. Using the full text of Shakespeare's *Hamlet*, the model learns to generate contextually relevant word suggestions.

---

## 🚀 Interactive Demo

Try the model live through the web interface:

👉 [next-word-model-with-hamlet.streamlit.app](https://next-word-model-with-hamlet.streamlit.app/)

---

## 🧠 Features

- **LSTM model training**: Trains a next-word prediction model on the complete text of *Hamlet*.
- **Streamlit interface**: Provides a simple and intuitive way to interact with the model.
- **Tokenization and preprocessing**: Includes essential steps to prepare data for training.
- **Model persistence**: Saves the trained model for later use.

---

## 🛠️ Project Structure

```plaintext
├── app.py                  # Streamlit app interface
├── experiments.ipynb       # Notebook with training and experiments
├── next_word_lstm.h5       # Trained LSTM model
├── requirements.txt        # Project dependencies
└── tokenizer.pickle        # Tokenizer used for preprocessing
