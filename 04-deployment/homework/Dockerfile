FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock /app/
RUN pipenv install --system --deploy 

COPY 'homework.py' '/app/homework.py'

ENTRYPOINT [ "python", "/app/homework.py", "-y 2023", "-m 05" ]