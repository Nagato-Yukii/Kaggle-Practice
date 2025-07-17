conda create -n map python=3.10.12 -y

conda activate map

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
