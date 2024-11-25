import sys
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Dropout, TextVectorization


class ModelService:
    def __init__(self, trainingFile):
        self.model = None
        self.text_vector = None
        self.model_training_file = trainingFile

    def predict_sentiment(self, text):
        text_to_analyze = [text]
        text_analyze_vector = self.text_vector(text_to_analyze)

        prediction = self.model.predict(text_analyze_vector)
        result = True
        confidence = prediction[0][0]

        if prediction[0][0] < 0.5:
            result = False
            confidence = 1 - prediction[0][0]

        print("result")
        print(result)
        print(confidence)

        return {
            "result": "Positive" if result else "Negative",
            "confidence": round(confidence * 100, 2),
        }

    def train_model(self):
        print("Pulling data from file...")
        base_dir = os.path.dirname(sys.prefix)
        file_path = os.path.join(base_dir, self.model_training_file)

        # Loading in the file continaing the preprocessed positive
        # and negative reviews
        dataFile = pd.read_csv(file_path)

        review_data = dataFile["review"].tolist()
        sentiment_values = dataFile["sentiment"].values

        print("Data Pulled Successfully!")

        # Encodes the Sentiment Values into numbers to work
        # with the neurons during model fitting
        print("Encoding Labels into numerals....")
        le = LabelEncoder()
        sentiment_encoded = le.fit_transform(sentiment_values)

        # vectorization for each review entry. This
        # converts each word into a number that can be
        # looked up and assessed during model fitting
        print("Creating Text Vectorization...")
        self.text_vector = TextVectorization(
            max_tokens=10000, output_mode="int", output_sequence_length=100
        )

        self.text_vector.adapt(review_data)

        reviews_vector_converted = self.text_vector(review_data)
        reviews_converted_np = reviews_vector_converted.numpy()

        # Using the sklearn module, we can provide our sentiments and reviews
        # and have traning and test data produced to provide to the model
        # during fitting
        print("Producing model test and training data...")
        reviews_train, reviews_test, sentiments_train, sentiments_test = (
            train_test_split(
                reviews_converted_np,
                sentiment_encoded,
                test_size=0.2,
                random_state=42,
            )
        )

        # This is defining the model as a Sequential model (meaning each layer
        # will be applied in order).
        print("Initializing model...")
        self.model = Sequential(
            [
                # Converts words into numerical representations
                Embedding(10000, 64),
                LSTM(64),  # Captures long-term dependencies between words
                # Introduces non-linearity, meaning the neural network can
                # learn complex patters and relationships within the data
                Dense(32, activation="relu"),
                # Prevents overfitting by randomly dropping out neurons during
                # training
                Dropout(0.2),
                # Output layer, showing probability between 0 and 1
                Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # This will kick off the training process, using the training data from
        # reviews and sentiments with the testing data generated in the split
        # above
        print("Starting model training...")
        self.model.fit(
            reviews_train,
            sentiments_train,
            epochs=1,
            validation_data=(reviews_test, sentiments_test),
        )

        print("Model trained and ready for usage!")
