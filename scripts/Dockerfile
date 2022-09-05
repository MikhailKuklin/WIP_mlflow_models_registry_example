FROM python:3.9

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "notebooks/models/model_log_reg.bin", "./" ]

EXPOSE 5000

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5000", "predict:app" ]