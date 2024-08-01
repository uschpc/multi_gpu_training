# Multi-GPU Training with PyTorch: Data and Model Parallelism

### About
The material in this repo demonstrates multi-GPU training using PyTorch. Part 1 covers how to optimize single-GPU training. The necessary code changes to enable multi-GPU training using the data-parallel and model-parallel approaches are then shown. This workshop aims to prepare researchers to use the new H100 GPU nodes as part of Princeton Language and Intelligence.

### Setup

Make sure you can run Python on Adroit:

```bash
$ ssh <YourNetID>@discovery.usc.edu  # VPN required if off-campus
$ git clone -b ITP-450 https://github.com/PrincetonUniversity/multi_gpu_training.git
$ cd multi_gpu_training
```

<!--
## Attendance

- Please check-in using [this link](https://cglink.me/2gi/c1471627125105938).

## Workshop Survey

Toward the end of the workshop please complete [this survey](https://forms.gle/pGi2tkzb7WCtVMcQ6).
-->

<!--
## Reminders

- The live workshop will be recorded
- Zoom: [https://princeton.zoom.us/my/picscieworkshop](https://princeton.zoom.us/my/picscieworkshop)
- Request an account on [Adroit](https://forms.rc.princeton.edu/registration/?q=adroit) if needed
- To use GPUs during the live workshop: `#SBATCH --reservation=multigpu`
-->


