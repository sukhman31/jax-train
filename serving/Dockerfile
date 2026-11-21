FROM python:3.12-slim

WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

COPY transformer/ transformer/
COPY tokenizers/ tokenizers/
COPY inference/ inference/
COPY lora/ lora/
COPY serving/ serving/

# Which trained checkpoint to bake in; set via [build.args] in fly.inference.toml
ARG CHECKPOINT=checkpoints/gpt_tinystories.pkl
COPY ${CHECKPOINT} checkpoints/model.pkl

ENV CHECKPOINT_PATH=checkpoints/model.pkl \
    GRPC_PORT=50051 \
    JAX_PLATFORMS=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 50051

CMD ["python", "-m", "serving.server"]
