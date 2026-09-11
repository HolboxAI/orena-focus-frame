#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="frame-algorithm"

if [ ! -f "${SCRIPT_DIR}/resources/model/config.json" ]; then
    echo "=+= resources/model/ is missing its merged model."
    echo "=+= Build it first:  python ../../scripts/merge_lora.py --base <base> --adapter <adapter> --out resources/model"
    exit 1
fi

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  "$SCRIPT_DIR" 2>&1
