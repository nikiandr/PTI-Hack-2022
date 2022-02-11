FROM python:3.10

WORKDIR /usr/src/app/

#   Install all requirements 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#   Setup kaggle
#
#   Environment variables for Kaggle:
#       KAGGLE_USERNAME - username in Kaggle
#       KAGGLE_TOKEN - Kaggle API token
#
RUN pip install kaggle
RUN mkdir /root/.kaggle
COPY kaggle.json /root/.kaggle

#   Copy data and program
COPY ./data ./data
COPY p.py .

CMD [ "python", "./p.py" ]