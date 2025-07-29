import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

def build_lstm_model(vocab_size, embedding_dim=100, input_length=200):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm(texts, labels, max_len=200, vocab_size=10000):
    # Tokenize
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)

    class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
    )

    class_weights = dict(enumerate(class_weights))

    # Train/test split
    
    X_train, X_test, y_train, y_test = train_test_split(padded, y_categorical, test_size=0.2, random_state=42)

    model = build_lstm_model(vocab_size, input_length=max_len)
    model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_split=0.1, callbacks=[early_stop],
          class_weight=class_weights)


    return model, tokenizer, le, X_test, y_test

def predict_lstm(model, X_test):
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    return y_pred
