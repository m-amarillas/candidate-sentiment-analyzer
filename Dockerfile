FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

# RUN python3 -m venv venv

# ENV VIRTUAL_ENV /venv
# ENV PATH /venv/bin:$PATH

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --ignore-installed

COPY . .

WORKDIR /app/src

CMD ["python3", "app.py"]

# CMD [ "gunicorn", "--worker-class" , "eventlet", "-w", "1", "--bind", "0.0.0.0:3000", "app:app"]