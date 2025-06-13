pip install -r  Requirements.txt
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the model
model = load_model('next_word_lstm.h5')

# Load the Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence:
        token_list = token_list[-(max_sequence-1):]# Ensure the sequence length matches
    token_list = pad_sequences([token_list],maxlen=max_sequence-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)

    for word ,index in tokenizer.word_index.items():
      if index == predicted_word_index:
        return word
    return None        

# Streamlit app
st.title("NEXT WORD PREDICTION")
input_text = st.text_input("ENTER THE WORD FOR THE PREDICTION","To be or not to be")
if st.button("Predict next word"):
  max_sequence =model.input_shape[1]+1
  next_word = predict_next_word(model,tokenizer,input_text,max_sequence)
  st.write(f'Next word :{next_word}')