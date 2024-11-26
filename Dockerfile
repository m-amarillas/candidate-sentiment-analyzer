FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --ignore-installed

COPY . .

WORKDIR /app/src

EXPOSE 3000

CMD [ "gunicorn", "--worker-class" , "eventlet", "-w", "1", "--bind", "0.0.0.0:3000", "app:app"]