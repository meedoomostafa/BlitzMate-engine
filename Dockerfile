FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt ./server-requirements.txt
RUN pip install --no-cache-dir -r server-requirements.txt

COPY engine/ ./engine/
COPY server/ ./server/
COPY setup_assets.py ./setup_assets.py
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

ENTRYPOINT ["./entrypoint.sh"]
