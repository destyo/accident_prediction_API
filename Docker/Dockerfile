FROM jupyter/scipy-notebook

WORKDIR /data

RUN pip install joblib
RUN pip install lightgbm
RUN pip install category_encoders
RUN pip install openpyxl
RUN pip install fastparquet

COPY NCDB_1999_to_2014.csv ./NCDB_1999_to_2014.csv

COPY predict.py ./predict.py
COPY train.py ./train.py

RUN python3 train.py
