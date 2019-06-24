FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

COPY requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY ./app /app