#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="frame-algorithm"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${ORENA_INPUT_DIR:-${SCRIPT_DIR}/test/input}"
OUTPUT_DIR="${ORENA_OUTPUT_DIR:-${SCRIPT_DIR}/test/output}"

# Use the GPU when Docker's NVIDIA runtime is available; fall back to CPU
# otherwise (the stub batch runs fine on CPU).
GPU_FLAGS=()
if docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia; then
    GPU_FLAGS=(--gpus all)
else
    echo "=+= No NVIDIA container runtime detected - running on CPU"
fi

# Rebuild only when the image is absent.
if ! docker image inspect "$DOCKER_IMAGE_TAG" > /dev/null 2>&1; then
    echo "=+= (Re)build the container"
    source "${SCRIPT_DIR}/do_build.sh"
fi

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true" \
      || true

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true
}

chmod -R -f o+rX "$INPUT_DIR"

if [ -d "${OUTPUT_DIR}/interface_1" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_1"
  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interface_1":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true" \
      || true
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_1"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    # '--network none' = no internet access, matching the platform.
    # '--volume <NAME>:/tmp' = the platform /tmp holds no permanent files.
    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        "${GPU_FLAGS[@]}" \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}

run_docker_forward_pass "interface_1"

echo "=+= Save this image for uploading via ./do_save.sh"
