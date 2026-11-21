#!/bin/bash
# Regenerate Python gRPC stubs from serving/protos. Run from anywhere: bash serving/generate_protos.sh
set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv-wsl/bin/python}

"$PYTHON" -m grpc_tools.protoc \
  -I serving/protos \
  --python_out=serving/gen \
  --grpc_python_out=serving/gen \
  serving/protos/inference/inference_api.proto

touch serving/gen/__init__.py serving/gen/inference/__init__.py
sed -i 's/^from inference import/from . import/' serving/gen/inference/inference_api_pb2_grpc.py

echo "Generated serving/gen/inference/"
