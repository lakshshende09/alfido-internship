FROM python:3.14

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn flask joblib

EXPOSE 5000

CMD ["python", "task3.py"]
