conda create -n kaggle python=3.10.12 -y

conda activate kaggle

pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
