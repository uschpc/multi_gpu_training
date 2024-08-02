# Multi-GPU Training with PyTorch: Data and Model Parallelism

### About
The material in this repo demonstrates multi-GPU training using PyTorch. Part 1 covers how to optimize single-GPU training. The necessary code changes to enable multi-GPU training using the data-parallel and model-parallel approaches are then shown. 

### Setup

Make sure you can run Python on Adroit:

```bash
$ ssh <YourNetID>@discovery.usc.edu  # VPN required if off-campus
$ git clone https://github.com/uschpc/multi_gpu_training.git
$ cd multi_gpu_training
```

### How was the Conda environment made?
Before you use Conda, make sure you have done the intitial setups of Conda following the link below: https://www.carc.usc.edu/user-guides/data-science/building-conda-environment
```bash
$ ssh <YourNetID>@discovery.usc.edu
$ salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=1:00:00
$ mamba create --name torch-env
$ mamba activate torch-env
$ mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ mamba install line_profiler --channel conda-forge




