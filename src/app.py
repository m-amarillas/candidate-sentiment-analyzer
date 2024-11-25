from flask import Flask, render_template
from flask_socketio import SocketIO
from services.model_service import ModelService

print("Starting app")

app = Flask(__name__)
socketio = SocketIO(app)

modelService = ModelService("MovieReviewTrainingDatabase.csv")
modelService.train_model()


@app.route("/")
def hello_world():
    return render_template("index.html")


@socketio.on("text_received")
def handle_text_received(text):
    if text is None:
        return False

    prediction = modelService.predict_sentiment(text)
    socketio.emit("prediction_ready", prediction)


if __name__ == "__main__":
    socketio.run(app, port=3000)
    print("App started. Listening...")
