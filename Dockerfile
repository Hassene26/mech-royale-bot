# Build from this folder: docker build -t my-bot .
FROM python:3.13-slim
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/main.py"]
