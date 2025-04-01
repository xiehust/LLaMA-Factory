#!/usr/bin/env bash
#sudo apt install -y python3.8-venv git
python3 -m venv /fsx/llamafactoryvenv
source /fsx/llamafactoryvenv/bin/activate
pip install -U pip

python3 -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
python3 -m pip install torch-neuronx==2.1.2.2.3.0 neuronx-cc==2.15.128.0 neuronx_distributed==0.9.0 torchvision
python3 -m pip install datasets transformers==4.46.2 peft huggingface_hub PyYAML trl==0.11.4 accelerate==0.29.2
python3 -m pip install git+https://github.com/huggingface/optimum-neuron.git
