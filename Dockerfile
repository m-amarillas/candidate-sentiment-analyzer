FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --ignore-installed

COPY . .

WORKDIR /app/src

EXPOSE 8080

CMD [ "gunicorn", "--worker-class" , "eventlet", "--bind", "0.0.0.0:8080", "app:app"]