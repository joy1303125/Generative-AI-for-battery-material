# Llama 3.1 8B Large Language Model

This is the Large Language Model (LLM) used in our paper to generate novel structures for TMO battery usage.

![Generated Crystals](assets/structures.png)

Updates are by [Amruth Nadimpally](https://github.com/amruthn1) and [Joy Datta](https://github.com/joy1303125)

## Try on Colab:
<a target="_blank" href="https://colab.research.google.com/github/joy1303125/Generative-AI-for-battery-material/blob/main/LLM_for_TMO_materials/LLM_for_TMO_materials.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<br/>

To setup the environment, run the following command:

### For ARM based Macs:

```
source install_arm.sh
```

### For CUDA based systems:

Go to https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation and edit the install_cuda.sh file with your appropriate Unsloth and CUDA version.

Then run:

```
source install_cuda.sh
```

### Train the model by running:

```
python llama_finetune.py --run-name 8b-run --model 8b
```

```
python llama_sample.py --model_name 8 --model_path ./exp/8b-run/
```

### To evaluate the results run:

```
python test_feasibility.py --output_dir ./outputs/llm_samples.csv
```
