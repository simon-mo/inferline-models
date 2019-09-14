#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

gcpprefix="gcr.io/clipper-model-comp"
tag="bench"

# Build RPC base images for python/anaconda and deep learning
# models
cd /home/ubuntu/clipper/model_composition/container_utils 
time docker build -t model-comp/tf-rpc -f TfRpcDockerfile ./
time docker build -t model-comp/tf-1-4-rpc -f Tf1-4RpcDockerfile ./

cd $DIR
# Build model-specific images
time docker build -t model-comp/tf-lang-detect -f TfLangDetectionDockerfile ./
docker tag model-comp/tf-lang-detect $gcpprefix/tf-lang-detect:$tag
# gcloud docker -- push $gcpprefix/tf-lang-detect:$tag

time docker build -t model-comp/tf-lstm -f TfLstmDockerfile ./
docker tag model-comp/tf-lstm $gcpprefix/tf-lstm:$tag
# gcloud docker -- push $gcpprefix/tf-lstm:$tag

time docker build -t model-comp/tf-nmt -f NmtDockerfile ./
docker tag model-comp/tf-nmt $gcpprefix/tf-nmt:$tag
# gcloud docker -- push $gcpprefix/tf-nmt:$tag
