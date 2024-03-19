FROM python:3.10-slim

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade streamlit==1.32.2 requests==2.31.0

COPY api.py .

CMD streamlit run api.py --server.port 8081