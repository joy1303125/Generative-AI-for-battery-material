CONDA_SUBDIR=osx-arm64 conda env create -f env.yml
conda activate crystal-llm

pip install torch torchvision torchaudio
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes pymatgen wandb
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0"

conda deactivate
