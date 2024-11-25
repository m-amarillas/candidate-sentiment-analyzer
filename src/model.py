import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Dropout, TextVectorization


# from sklearn.model_selection import train_test_split


# from tensorflow.keras.preprocessing.sequence import pad_sequences

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def load_file():
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    print("loading file...")
    dataFile = pd.read_csv("MovieReviewTrainingDatabase.csv")
    dataFile["review"] = dataFile["review"].apply(preprocess_text)

    dataFile.to_csv("preprocessed_text.csv", index=False)


# Preprocess text
def preprocess_text():
    dataFile = pd.read_csv("preprocessed_text.csv")
    words = dataFile["review"].apply(word_tokenize)
    words = [word for word in words if word not in stopwords.words("english")]
    combined_words = list(itertools.chain.from_iterable(words))

    processedWords = " ".join(combined_words)

    with open("preprocessed_text.txt", "w") as wordsFile:
        wordsFile.write(processedWords)


def tokenize_text():
    dataFile = pd.read_csv("MovieReviewTrainingDatabase.csv")

    text_data = dataFile["review"].tolist()
    y = dataFile["sentiment"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(y_encoded)

    text_vector = TextVectorization(
        max_tokens=10000, output_mode="int", output_sequence_length=100
    )

    text_vector.adapt(text_data)

    X = text_vector(text_data)
    x_np = X.numpy()

    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        x_np, y_encoded, test_size=0.2, random_state=42
    )

    print("train test split done")

    print("creating model...")
    model = Sequential(
        [
            Embedding(10000, 64),
            LSTM(64),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),  # Assuming 3 sentiment classes
        ]
    )

    print("compiling model...")
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    print("fitting model...")
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    print("model trained!")

    print("Testing Model with positive text")
    test_text = ["This Movie is Fantastic!"]
    test_text_vector = text_vector(test_text)
    prediction = model.predict(test_text_vector)

    print(prediction)

    print("Testing Model with negative text")
    test_text2 = ["This movie was not great"]
    test_text_vector2 = text_vector(test_text2)
    prediction2 = model.predict(test_text_vector2)

    print(prediction2)

    print("complete")
