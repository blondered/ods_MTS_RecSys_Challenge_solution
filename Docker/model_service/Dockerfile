FROM python:3.8.13

#
WORKDIR /code

#
RUN pip install --upgrade pip
RUN pip install poetry==1.1.15

#
COPY ./pyproject.toml ./poetry.lock /code/

COPY ../../src/app_backend /code/app_backend
COPY ./Docker/model_service/.env /code/app_backend/.env


RUN poetry config virtualenvs.create false \
  && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi
  
  
CMD ["uvicorn", "app_backend.app:app", "--host", "0.0.0.0", "--port", "80"]
