from services.model_service import ModelService
from flask import Flask, render_template
from flask_socketio import SocketIO

print("Starting app...")

app = Flask(__name__)
socketio = SocketIO(app)

modelService = ModelService("MovieReviewTrainingDatabase.csv", socketio)


@app.route("/")
def main():
    return render_template("index.html")


@socketio.on("load_model")
def load_model():
    socketio.emit("model-status-update", "Training Model...")

    model_stats = modelService.train_model()

    socketio.emit("model-loaded", modelService.format_model_stats(model_stats))


@socketio.on("text_received")
def handle_text_received(text):
    if text is None:
        return False

    prediction = modelService.predict_sentiment(text)
    socketio.emit("prediction_ready", prediction)


if __name__ == "__main__":
    print("App started. Listening...")
    socketio.run(app, port=3000)
