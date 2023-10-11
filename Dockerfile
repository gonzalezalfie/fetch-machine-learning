FROM ubuntu:22.04

WORKDIR /home/ubuntu

ADD data_daily.csv .

ADD model.py .

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib==3.7.0 \
    seaborn==0.12.2 \
    --no-cache-dir

CMD ["python3", "./model.py"]


