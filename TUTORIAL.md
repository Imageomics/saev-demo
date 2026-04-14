# SAEs for Discovery (Imageomics Conference Tooling Workshop)

This repository is designed for use in the Imageomics Conference Tooling Workshop. This file details the exact process used to extract vision model activations, train SAEs, and visualize the results. This process will be outlined in the tooling demo and the live iteration discovery workshop.

Note that this document details the entire process used for generating visualizations/SAEs for the Imageomics Conference Tooling Workshop SAE demo. The directories for saving/loading artifacts are going to be specific to the computing cluster and user used for this process, and should be changed to reflect your system.

## Downloading datasets

Three datasets are used for visuals in this workshop:

- [Fish-Vista](https://huggingface.co/datasets/imageomics/fish-vista) - A dataset of fish specimen with species and trait labels.
- [2018 NEON Beetles Dataset](https://huggingface.co/datasets/imageomics/2018-NEON-beetles) - A dataset of beetle specimens collected from NEON sites in 2018 and imaged in 2021
- [Cambridge Heliconius Butterfly Collection](https://huggingface.co/datasets/imageomics/Heliconius-Collection_Cambridge-Butterfly) - A dataset of Heliconius butterfly specimens spanning several mimic subspecies pairs. 

Each dataset can be downloaded using the directions on their respective huggingface pages. For the demo/tutorial, we predominantly make use of the iNaturalist 2021 Mini dataset, which can be downloaded and extracted using the following commands:

`wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz`
`tar -xvzf train_mini.tar.gz`

## Saving Vision Model Activations

Vision model activations can be easily generated using the saev package, which this repo is a fork of. In this workshop, we use DINOv2 ViT-S model. For this model, we save the activations (dim 384) from the second to last layer. This can be done from running the following command from the root directory of this repository:

### Fist-Vista
`uv run launch.py shards --shards-root /local/scratch/beattie.74/saev_demo/saev/shards --family dinov2 --ckpt dinov2_vits14_reg --d-model 384 data:img-folder --data.root /local/scratch/beattie.74/fish_sae/fish-vista/shard_input`

These shards were saved into `/local/scratch/beattie.74/saev_demo/saev/shards/02b05f61`

### Beetles
`uv run launch.py shards --shards-root /local/scratch/beattie.74/saev_demo/saev/shards --family dinov2 --ckpt dinov2_vits14_reg --d-model 384 data:img-folder --data.root /local/scratch/beattie.74/saev_demo/2018-NEON-beetles/individual_specimens/`

These shards were saved into `/local/scratch/beattie.74/saev_demo/saev/shards/1dab97bf`


### Heliconius Butterflies
`uv run launch.py shards --shards-root /local/scratch/beattie.74/saev_demo/saev/shards --family dinov2 --ckpt dinov2_vits14_reg --d-model 384 data:img-folder --data.root /local/scratch/beattie.74/saev_demo/Heliconius_butterflies/`

These shards were saved into `/local/scratch/beattie.74/saev_demo/saev/shards/b7c18cf7`

### iNaturalist 2021 Mini: 
`uv run launch.py shards --shards-root /local/scratch/beattie.74/saev_demo/saev/shards --family dinov2 --ckpt dinov2_vits14_reg --d-model 384 data:img-folder --data.root /local/scratch/beattie.74/inat/train_mini`

These shards were saved into `/local/scratch/beattie.74/saev_demo/saev/shards/73e0f5b8`

## Training SAEs

We train separate SAEs for each dataset. Typically, a hyperparameter sweep is done in order to train an SAE with optimal hyperparameters for learning meaningful traits. Here, we sweep on learning rate and k. We use a Matryoshka SAE with 10 prefixes per batch, 16x larger sparse dimension, and BatchTopK activation function:

### Fish-Vista
`uv run launch.py train --sweep ./demo/demo_sweep.py --mem-gb 48 --n-train 50000000 --n-val 10 --runs-root /local/scratch/beattie.74/saev_demo/saev/runs --train-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/02b05f61/ --train-data.layer -2 --val-data.layer -2 --val-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/02b05f61/ --sae.d-model 384 --sae.d-sae 6144 --objective.dead-threshold-tokens 1000000 sae.activation:batch-top-k sae.activation.sparsity:no-sparsity`

Moved run files to `/local/scratch/beattie.74/saev_demo/fish/saev/runs`


### Beetles
`uv run launch.py train --sweep ./demo/demo_sweep.py --mem-gb 48 --n-train 50000000 --n-val 10 --runs-root /local/scratch/beattie.74/saev_demo/saev/runs --train-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/1dab97bf/ --train-data.layer -2 --val-data.layer -2 --val-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/1dab97bf/ --sae.d-model 384 --sae.d-sae 6144 --objective.dead-threshold-tokens 1000000 sae.activation:batch-top-k sae.activation.sparsity:no-sparsity`

Moved run files to `/local/scratch/beattie.74/saev_demo/beetles/saev/runs`


### Butterflies
`uv run launch.py train --sweep ./demo/demo_sweep.py --mem-gb 48 --n-train 50000000 --n-val 10 --runs-root /local/scratch/beattie.74/saev_demo/saev/runs --train-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/b7c18cf7/ --train-data.layer -2 --val-data.layer -2 --val-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/b7c18cf7/ --sae.d-model 384 --sae.d-sae 6144 --objective.dead-threshold-tokens 1000000 sae.activation:batch-top-k sae.activation.sparsity:no-sparsity`

Moved run files to `/local/scratch/beattie.74/saev_demo/butterflies/saev/runs`

### iNaturalist 2021 Mini
`uv run launch.py train --sweep ./demo/demo_sweep.py --mem-gb 48 --n-train 50000000 --n-val 10 --runs-root /local/scratch/beattie.74/saev_demo/saev/runs --train-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/73e0f5b8/ --train-data.layer -2 --val-data.layer -2 --val-data.shards /local/scratch/beattie.74/saev_demo/saev/shards/73e0f5b8/ --sae.d-model 384 --sae.d-sae 6144 --objective.dead-threshold-tokens 1000000 sae.activation:batch-top-k sae.activation.sparsity:no-sparsity`

Runs located in `/local/scratch/beattie.74/saev_demo/saev/runs/`

## Pareto Analysis of SAE Runs

For a specific dataset, we do not know what hyperparameters are going to be best for our SAE training. To address this, we sweep over possible hyperparameters (specifically learning rate and k for BatchTopK activation function). These SAEs are then plotted on a mean squared error (MSE) vs k plot, and the models on the *pareto frontier* (best MSE/k tradeoff) are selected.

Use `demo/pareto_runs.py` to run a Pareto analysis over SAE runs with:
- x-axis: BatchTopK `k` (used as L0 sparsity proxy)
- y-axis: average MSE (from WandB summary, preferring `eval/mse`)

To run the pareto analysis for the iNaturalist 2021 Mini training run, use the following command:
`uv run python demo/pareto_runs.py --runs-root /local/scratch/beattie.74/saev_demo/saev/runs --out-png demo/pareto_k_vs_mse.png --out-csv demo/pareto_k_vs_mse.csv --annotate`

This writes:
- Plot: `demo/pareto_k_vs_mse.png`, a graphical representation of the pareto frontier, with the Pareto frontier highlighted in red.
- Table: `demo/pareto_k_vs_mse.csv`, a CSV containing the data used for plotting the above figure.

Choose one of the SAEs on the Pareto frontier to use for inference/visualization.

## Running Inference (Generating Sparse Features)

To identify what images maximally activate each feature, we need to generate sparse embeddings for every image in the dataset. This can be done using the inference interface:

`uv run launch.py inference --run /local/scratch/beattie.74/saev_demo/saev/runs/p7zyxu5e/ --data.shards /local/scratch/beattie.74/saev_demo/saev/shards/73e0f5b8/`

For this demo, this is done for the iNaturalist 2021 Mini dataset.

## Visualizing Fish SAE Features

Finally, we can visualize SAE features on images from our dataset. `demo/fish_feature_vis.py` is a Streamlit app for visualizing fish SAE features from inference outputs.

This app:
- Uses run `/local/scratch/beattie.74/saev_demo/fish/saev/runs/5nprxikb` by default.
- Samples a number of images from the specified dataset at random to generate sparse features for.
- For each feature, finds the top 10 images with highest sparse activation from this randomly selected dataset.
- Recomputes SAE feature maps on those DINO activations and overlays per-patch activations on each image.

Run the app:
`uv run --with streamlit streamlit run demo/fish_feature_vis.py`

If `streamlit` is already installed in your environment:
`uv run streamlit run demo/fish_feature_vis.py`
