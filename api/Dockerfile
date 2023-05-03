FROM jbei/scikit-learn

WORKDIR /app

COPY requirements.txt /app/
COPY ./app /app

RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip && \
    pip install --no-cache-dir fastapi pydantic datetime pickle5 uvicorn clearml joblib

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
