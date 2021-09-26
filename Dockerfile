FROM python:3.6
WORKDIR /Image-Super-Resolution
COPY . .
RUN pip3 install requirements.txt

