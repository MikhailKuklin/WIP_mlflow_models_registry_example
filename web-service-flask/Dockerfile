FROM python:3.9-slim-buster

RUN pip install -U pip

ARG WORKDIR=/app
WORKDIR ${WORKDIR}

COPY [ "../requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY [ "predict.py", "notebooks/models/model_log_reg.bin", "./" ]

EXPOSE 5000

#ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5000", "predict:${WORKDIR}" ]
