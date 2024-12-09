# Candidate Sentiment Analyzer
The Sentiment Analyzer will take recorded interview conversational text and interpret the candidate's sentiment. When paired with the AI voice conversation capture, you can begin interpreting the candidate's tone and personality. Additionally, you can match the sentiment with the position's requirements.

For example, if you're hiring for a customer-facing role, you want somebody who is more upbeat and positive. If you seek a leadership/executive role, you want somebody concise yet strategic. This app is the starting point for providing these insights using model-building APIs from TensorFlow.

## Tech Specs

* TensorFlow APIs are used for faster model training, testing, and prediction
* Model Training CSV Included with 25,000 movie reviews, categorized with Positive or Negative ratings
* Flask and Flask_SocketIO are used for the build server and front end communication
* Front end is vanilla HTML,CSS, and Javascript, connected to backend via SocketIO


## Build Commands

1. Create the enviornment using this command:

    ```python3 -m venv venv```

    ```source venv/bin/activate```

2. Install the Requirements

    ```pip install -r requirements.txt```

## Run theApp

1. If not done, run source venv/bine/activate to enter the virtual machine
2. Navigate to src ```cd src```
2. Run the app in development mode
    
    ```python3 app.py```
2. Run the app in production mode

    ```gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:3000 app:app```

## Build and Run Docker

1. Run the docker file

    ```docker build -t sentiment-analysis .```
2. Start the Image

    ```docker run -d -p 80:5000 sentiment-analysis```