# streamlit_app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Input, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
                                     LSTM, Attention, Dense, Dropout, GlobalAveragePooling2D, Concatenate)
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from PIL import Image

st.title("🧠 Multi-Modal Deep Learning Model")

# --- Image Upload ---
st.header("📷 Image Classification")
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_img:
    img = Image.open(uploaded_img).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

# --- Text Input ---
st.header("💬 Text Sentiment Analysis")
text_input = st.text_area("Enter a review or tweet")
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts([text_input])
seq = tokenizer.texts_to_sequences([text_input])
text_array = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

# --- Time Series Input ---
st.header("📈 Time Series Forecasting")
ts_csv = st.file_uploader("Upload time series CSV", type="csv")
if ts_csv:
    df_ts = pd.read_csv(ts_csv)
    ts_data = df_ts.values
    ts_data = (ts_data - np.min(ts_data)) / (np.max(ts_data) - np.min(ts_data))
    ts_seq = ts_data[:100].reshape(1, 100, 1)

# --- Model Definition ---
def build_model():
    # Image branch
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_cnn = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    x_img = GlobalAveragePooling2D()(base_cnn.output)
    x_img = Dropout(0.3)(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    img_out = Dense(10, activation='softmax', name='img_out')(x_img)

    # Text branch
    text_input = Input(shape=(100,), name='text_input')
    x_txt = Embedding(input_dim=10000, output_dim=128)(text_input)
    attn_txt = MultiHeadAttention(num_heads=4, key_dim=128)(x_txt, x_txt)
    x_txt = LayerNormalization()(x_txt + attn_txt)
    x_txt = GlobalAveragePooling1D()(x_txt)
    x_txt = Dense(256, activation='relu')(x_txt)
    txt_out = Dense(3, activation='softmax', name='txt_out')(x_txt)

    # Time series branch
    ts_input = Input(shape=(100, 1), name='ts_input')
    x_ts = LSTM(128, return_sequences=True)(ts_input)
    attn_ts = Attention()([x_ts, x_ts])
    x_ts = Dense(64, activation='relu')(attn_ts)
    ts_out = Dense(1, activation='linear', name='ts_out')(x_ts)

    model = Model(inputs=[image_input, text_input, ts_input],
                  outputs=[img_out, txt_out, ts_out])
    return model

model = build_model()
model.compile(optimizer='adam',
              loss={'img_out': 'categorical_crossentropy',
                    'txt_out': 'categorical_crossentropy',
                    'ts_out': 'mse'},
              metrics={'img_out': 'accuracy',
                       'txt_out': 'accuracy',
                       'ts_out': 'mae'})

# --- Prediction ---
if st.button("Run Prediction"):
    inputs = {}
    if uploaded_img:
        inputs['image_input'] = img_array
    if text_input:
        inputs['text_input'] = text_array
    if ts_csv:
        inputs['ts_input'] = ts_seq

    dummy_img = np.zeros((1, 10))
    dummy_txt = np.zeros((1, 3))
    dummy_ts = np.zeros((1, 1))

    preds = model.predict(inputs)
    st.success("✅ Prediction Complete")
    st.write("Image Classification Output:", preds[0])
    st.write("Text Sentiment Output:", preds[1])
    st.write("Time Series Forecast Output:", preds[2])
