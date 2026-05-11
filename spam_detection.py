import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LSTM,
    Dropout
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

print("\nLoading Dataset...\n")

df = pd.read_csv("C:/Users/Shiva Itchapurapu/Desktop/spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

encoder = LabelEncoder()

df['label'] = encoder.fit_transform(df['label'])

X = df['message']
y = df['label']

print("\nTokenizing Text...\n")

vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size)

tokenizer.fit_on_texts(X)

X_sequences = tokenizer.texts_to_sequences(X)

max_length = 100

X_padded = pad_sequences(
    X_sequences,
    maxlen=max_length
)

X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nBuilding Model...\n")

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=max_length
    )
)

model.add(LSTM(64))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

print("\nTraining Model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")
print(report)

with open("outputs/reports/classification_report.txt", "w") as f:
    f.write(report)

model.save("outputs/model/spam_detection_model.h5")

print("\nModel Saved Successfully!")

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig("outputs/graphs/accuracy_graph.png")

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("outputs/graphs/loss_graph.png")

plt.close()

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/graphs/confusion_matrix.png")

plt.close()

def predict_spam(text):

    text_seq = tokenizer.texts_to_sequences([text])

    text_pad = pad_sequences(
        text_seq,
        maxlen=max_length
    )

    prediction = model.predict(text_pad)

    spam_probability = prediction[0][0]

    print("\nMessage:", text)
    print("Spam Probability:", spam_probability)

    if spam_probability > 0.5:
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")

print("\nTesting New Messages...\n")

predict_spam("Congratulations! You won a free iPhone")

predict_spam("Hi brother, where are you now?")

predict_spam("URGENT! Claim your reward now")

predict_spam("Let's meet tomorrow for lunch")

print("\nProject Completed Successfully!")

print("\nSaved Outputs:")
print("1. Model")
print("2. Accuracy Graph")
print("3. Loss Graph")
print("4. Confusion Matrix Graph")
print("5. Classification Report")