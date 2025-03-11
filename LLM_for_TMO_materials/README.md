# Llama 3.1 8B Large Language Model

This is the Large Language Model (LLM) used in our paper to generate novel structures for TMO battery usage.

![Generated Crystals](assets/structures.png)

Updates are by [Amruth Nadimpally](https://github.com/amruthn1) and [Joy Datta](https://github.com/joy1303125)

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
python llama_finetune.py --run-name 7b-test-run --model 7b
```

### To evaluate the results run:

```
python test_feasibility.py --output_dir ./path/to/your/generated/cif/files
```
