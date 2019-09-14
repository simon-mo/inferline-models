#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

prefix="gcr.io/clipper-model-comp"
tag="bench"

#time docker build -t model-comp/py-rpc -f RpcDockerfile ./
#time docker build -t model-comp/cuda-rpc -f CudaPyRpcDockerfile ./
#time docker build -t model-comp/tf-rpc-uncompiled -f TfRpcDockerfile ./
# docker tag model-comp/tf-rpc-uncompiled model-comp/tf-rpc

#tf_sha=$(nvidia-docker run -d model-comp/tf-rpc-uncompiled)

#d=$(docker ps -q | wc -l)

#while [ $d -ne "0" ]; do
#        sleep 3
#        d=$(docker ps -q | wc -l)
#        docker ps
#done

#docker container commit $tf_sha model-comp/tf-rpc

#time docker build -t $prefix/pytorch:$tag -f PyTorchDockerfile ./

#time docker build -t $prefix/pytorch-preprocess:$tag -f PreprocessPyTorchDockerfile ./


declare -a models=(
        "alexnet"
        #"vgg11"
        #"vgg13"
        #"vgg16"
        #"vgg19"
        "res18"
        "res34"
        "res50"
        "res101"
        "res152"
        "squeezenet10"
        "squeezenet11"
        #"densenet121"
        #"densenet169"
        #"densenet161"
        #"densenet201"
        # "inceptionv3"
        )

for m in "${models[@]}"
do
     echo "Skipping $m"
#    time docker build --build-arg MODEL="$m" -t $prefix/pytorch-$m:$tag -f ModelPyTorchDockerfile ./
done

# Build model-specific images
#time docker build -t $prefix/tf-kernel-svm:$tag -f TfKernelSvmDockerfile ./
#time docker build -t $prefix/tf-resnet-feats:$tag -f TfResNetDockerfile ./
#time docker build -t $prefix/tf-resnet-feats-variable-input:$tag -f TfResNetVariableInputSizeDockerfile ./
#time docker build -t $prefix/tf-log-reg:$tag -f TfLogisticRegressionDockerfile ./
#time docker build -t $prefix/inception-feats:$tag -f TfInceptionFeaturizationDockerfile ./

#time docker build -t $prefix/tf-nmt:$tag -f TfNMTDockerfile ./
time docker build -t $prefix/tf-lang-detect:$tag -f TfLangDetectionDockerfile ./

#time docker build -t $prefix/pytorch-yolo:$tag -f PyTorchYoloDockerfile ./
#time docker build -t $prefix/alpr:$tag -f ALPRDockerfile ./
