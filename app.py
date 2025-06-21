import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

window_size = model.input_shape[1]

# Function to predict the next worddef generate_text_sequence(model, tokenizer, seed_text, num_words=10):
def predict_next_word(model, tokenizer, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=window_size, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    return tokenizer.index_word.get(predicted_word_index, "<OOV>")

# streamlit app
st.title("Next Word Prediction")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"): # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text,)
    st.write(f'Next word: {next_word}')