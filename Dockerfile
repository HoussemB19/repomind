FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout 400 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 400 --retries 5 -r requirements.txt

COPY hf_model_cache/ /root/.cache/huggingface/hub/
ENV HF_HUB_OFFLINE=1

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]