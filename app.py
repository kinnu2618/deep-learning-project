import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Custom styles for black theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    .stTextInput textarea {
        background-color: #2c2f33;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ffffff;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 8px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #ff6b6b;
    }
    h3 {
        color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title('IMDB Sentiment Analysis Tool')
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg', width=200)
st.sidebar.write('Use this tool to classify movie reviews as **positive**, **average**, or **negative** based on sentiment analysis.')

# Main UI
st.title('🎬 IMDB Movie Review Sentiment Classifier 🎥')
st.markdown('Enter a movie review below, and this tool will classify it based on sentiment analysis using a pre-trained model.')

# About section
with st.expander("ℹ️ About this App"):
    st.write("This app uses a pre-trained model on the IMDB dataset to classify the sentiment of movie reviews as positive, average, or negative.The IMDB Movie Review Sentiment Classifier is a web-based application that allows users to input movie reviews and analyze the sentiment using a pre-trained deep learning model. The model classifies the review as Positive, Average, or Negative, and provides a confidence score to indicate how confident the model is about its prediction.")

# User input in text area
user_input = st.text_area('📝 Enter Your Movie Review Here:')

# Sentiment classification button
if st.button('🔍 Analyze Sentiment'):
    if user_input:
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]

        # Determine sentiment based on the score
        if score > 0.75:
            sentiment = '😄 Positive'
            sentiment_color = '#00ff00'  # Green
        elif 0.61 <= score <= 0.75:
            sentiment = '😐 Average'
            sentiment_color = '#ffcc00'  # Yellow
        else:
            sentiment = '😞 Negative'
            sentiment_color = '#ff4b4b'  # Red

        # Display the result with color-coded sentiment
        st.subheader('Sentiment Result:')
        st.markdown(f'<h3 style="color:{sentiment_color};">{sentiment}</h3>', unsafe_allow_html=True)
        st.write(f'Prediction Confidence: **{score:.2f}**')
    else:
        st.warning('Please enter a movie review to classify!')
else:
    st.write('Awaiting your input...')

# Footer
st.sidebar.write('---')
st.sidebar.write("© 2024 IMDB Sentiment Analysis App")
