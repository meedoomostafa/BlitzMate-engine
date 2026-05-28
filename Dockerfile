FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt ./server-requirements.txt
RUN pip install --no-cache-dir -r server-requirements.txt

COPY engine/ ./engine/
COPY server/ ./server/

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
