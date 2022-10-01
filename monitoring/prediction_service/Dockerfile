FROM python:3.9.7-slim-buster

# install pip to the image
RUN pip install -U pip

# define workdir 
WORKDIR /app

# copy and install requirements
COPY [ "../requirements.txt", "./" ]

RUN pip install -r requirements.txt

# copy necessary files which will be used
COPY [ "predict.py", "model_log_reg.bin", "./" ]

# define port
EXPOSE 6000

# instuction on what should be run when a contained start
# from the docker image
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:6000", "predict:app" ]
