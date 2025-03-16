## Go to https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation to figure out the proper CUDA and Unsloth version for your system
## Then replace the Unsloth install line with your proper Unsloth+CUDA version

conda env create -f env.yml
conda activate crystal-llm

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes pymatgen wandb
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0"
pip install mp_api ase matgl

conda deactivate
