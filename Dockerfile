FROM python:3.6
WORKDIR /Image-Super-Resolution
COPY . /app
RUN pip3 install requirements.txt
EXPOSE  5000
CMD ["python", "app.py"]