FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

COPY api.py .

RUN mkdir -p ./utils
COPY ./utils/lm.py ./utils/lm.py
COPY ./utils/web_search.py ./utils/web_search.py

CMD uvicorn api:app --host 0.0.0.0 --port 8080