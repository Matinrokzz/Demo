FROM ubuntu:18.04

LABEL Name=salespredictionsiriustechendpoint Version=0.0.1

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt install software-properties-common -y 
RUN add-apt-repository ppa:deadsnakes/ppa -y 

# Python package management and basic dependencies
RUN apt install python3.7 -y
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7
RUN apt install python3-pip -y

RUN python -m pip install --upgrade pip
RUN apt-get install -y curl

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


#RUN useradd appuser && chown -R appuser /app
#USER appuser
# Install pip requirements
RUN pip install numpy
RUN pip install Flask
RUN pip install flask
RUN pip install sklearn
RUN pip install joblib
RUN pip install seaborn
RUN pip install datetime
RUN pip install pandas
RUN pip install catboost
RUN pip install xgboost
RUN pip install pickle
RUN pip install warnings
RUN pip install matplotlib
RUN pip install scikit-build
RUN pip install scikit-learn


WORKDIR /
COPY . /

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug

CMD ["python", "endpoint.py"]