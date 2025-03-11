conda create -y --name cdvae python==3.8

conda activate cdvae

pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -e .
pip install "cython<3.0.0" wheel
pip install "pyyaml==5.4.1" --no-build-isolation
pip install omegaconf pytorch_lightning==1.3.8 hydra-core==1.1.0 python-dotenv torch_geometric==2.1.0 pandas pymatgen==2023.8.10 p_tqdm torchmetrics==0.5 wandb mp-api

if [ -e .env ]
then
    echo ".env exists"
else
    touch .env
    echo "export PROJECT_ROOT='$(pwd)'" > .env
    echo "export HYDRA_JOBS='$(pwd)/hydra'" >> .env
    echo "export WABDB_DIR='$(pwd)/cdvae/wabdb'" >> .env
    echo "Go to https://wandb.ai/authorize and get an API key. Enter the API Key below: "
    read apikey
    echo "export WANDB_API_KEY='$apikey'" >> .env
    echo "Go to https://next-gen.materialsproject.org/api and get an API key. Enter the API Key below: "
    read mpapikey
    echo "export MP_API_KEY='$mpapikey'" >> .env
fi
