FROM python

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]