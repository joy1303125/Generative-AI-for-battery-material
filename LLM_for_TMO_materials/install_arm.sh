CONDA_SUBDIR=osx-arm64 conda env create -f env.yml
conda activate crystal-llm

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes pymatgen wandb
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0"
pip install mp_api ase matgl

conda deactivate
