FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/agent.py .
COPY src/llm_utils.py .
COPY src/main.py .
COPY src/models.py .
COPY src/prompts.py .
COPY src/prompt.txt .
COPY src/tools.py .
COPY src/variables.py .
COPY src/vectorstore.py .

COPY refdata /refdata

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]