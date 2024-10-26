FROM quay.io/astronomer/astro-runtime:12.2.0

COPY requirements.txt /requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r /requirements.txt
# RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install openpyxl
