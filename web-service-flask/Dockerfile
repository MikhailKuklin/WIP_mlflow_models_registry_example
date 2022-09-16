FROM python:3.9.7-slim-buster

RUN pip install -U pip

ARG WORKDIR=/app
WORKDIR ${WORKDIR}

COPY [ "../requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY [ "predict.py", "model_log_reg.bin", "./" ]

EXPOSE 6000

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:6000", "predict:app" ]
