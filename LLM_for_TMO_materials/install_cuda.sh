## Go to https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation to figure out the proper CUDA and Unsloth version for your system
## Then replace the Unsloth install line with your proper Unsloth+CUDA version

conda env create -f env.yml
conda activate crystal-llm

pip install torch torchvision torchaudio
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes pymatgen wandb
pip install "unsloth[cu118-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0"

conda deactivate
