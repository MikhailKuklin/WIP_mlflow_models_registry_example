# Example of deployment of a machine-learning model as a web service 
using:
- Docker :whale:
- Flask

## Content

*predict.py*: script contains necessary functions and command to run using Flask

*test.py*: script contains feature information based on which prediction is made and result is 
printed

*Dockerfile*: docker file for creating a docker image to run prediction in a container i.e. one 
does not need to install requirements separately

*model_log_reg.bin*: model obtained from previous steps and copied to the folder

*requirements.txt*: necessary packages

## How to run?
1. Build Docker image:
docker build -t sex-penguins-prediction:v1 .

2. Run 
docker run -it --rm -p 6000:6000 sex-penguins-prediction:v1
