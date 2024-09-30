conda create -n crystal-llm
conda activate crystal-llm

pip install torch torchvision torchaudio
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes pymatgen wandb mp_api ase pandas argparse

conda deactivate