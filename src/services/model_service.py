import os
import pandas as pd
import keras as kr
import pickle
from enum import Enum


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Dropout, TextVectorization

TRAINING_PASSES = 10
MODEL_NAME = "SENTIMENT_MODEL.keras"
MODEL_STATS_FILE = "MODEL_STATS.pk1"


class ModelStats(Enum):
    Accuracy = "accuracy"
    Loss = "loss"
    Val_Accuracy = "val_accuracy"
    Val_Loss = "val_loss"


class ModelService:
    def __init__(self, trainingFile, socket):
        if os.path.exists(MODEL_NAME):
            self.model = kr.models.load_model(MODEL_NAME)
        else:
            self.model = None

        print("Pulling data from file...")

        current_dir = os.path.dirname(__file__)
        src_dir = os.path.dirname(current_dir)
        base_dir = os.path.dirname(src_dir)
        file_path = os.path.join(base_dir, trainingFile)

        # Loading in the file continaing the preprocessed positive
        # and negative reviews
        dataFile = pd.read_csv(file_path)

        self.review_data = dataFile["review"].tolist()
        self.sentiment_values = dataFile["sentiment"].values

        print("Data Pulled Successfully!")

        # vectorization for each review entry. This
        # converts each word into a number that can be
        # looked up and assessed during model fitting
        print("Creating Text Vectorization...")
        self.text_vector = TextVectorization(
            max_tokens=10000, output_mode="int", output_sequence_length=100
        )

        self.text_vector.adapt(self.review_data)

        self.socket = socket

    def predict_sentiment(self, text):
        text_to_analyze = [text]
        text_analyze_vector = self.text_vector(text_to_analyze)

        prediction = self.model.predict(text_analyze_vector)
        result = True
        confidence = prediction[0][0]

        if confidence < 0.75:
            result = False
            confidence = 1 - confidence

        return {
            "result": "Positive" if result else "Negative",
            "confidence": round(confidence * 100, 2),
        }

    def train_model(self):
        if self.model is not None:
            with open(MODEL_STATS_FILE, "rb") as file:
                model_stats = pickle.load(file)

            return model_stats

        # Encodes the Sentiment Values into numbers to work
        # with the neurons during model fitting
        print("Encoding Labels into numerals....")
        le = LabelEncoder()
        sentiment_encoded = le.fit_transform(self.sentiment_values)

        reviews_vector_converted = self.text_vector(self.review_data)
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
        model_history = self.model.fit(
            reviews_train,
            sentiments_train,
            epochs=TRAINING_PASSES,
            validation_data=(reviews_test, sentiments_test),
        )

        with open(MODEL_STATS_FILE, "wb") as file:
            pickle.dump(model_history.history, file)

        self.model.save(MODEL_NAME)

        print("Model trained and ready for usage!")

        return model_history.history

    def format_model_stats(self, model_stats):
        stats = f"Training Passes: {TRAINING_PASSES}"
        for stat in ModelStats:
            statValueArray = model_stats[stat.value]
            statValue = statValueArray[len(statValueArray) - 1]
            stats += f" - {stat.name}: {statValue: .2f}"

        return stats
