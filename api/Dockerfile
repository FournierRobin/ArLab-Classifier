FROM python:3.10.11-alpine

WORKDIR /app

COPY requirements.txt /app/
COPY ./app /app

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev \
    libffi-dev openssl-dev py3-numpy g++ && \
    apk add --no-cache openblas-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
