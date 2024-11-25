## Build Commands

1. Create the enviornment using this command:

    ```python3 -m venv venv```

    ```source venv/bin/activate```

2. Install the Requirements

    ```pip install -r requirements.txt```

## Run theApp

1. If not done, run source venv/bine/activate to enter the virtual machine
2. Run the app

    ```python3 -m flask run```

## Build and Run Docker

1. Run the docker file

    ```docker build -t sentiment-analysis .```
2. Start the Image

    ```docker run -d -p 80:5000 sentiment-analysis```