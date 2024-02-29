# Evaluating Conformal Prediction Under Distribution Shift

This repository contains code for the paper [**Empirically Validating Conformal Prediction on Modern Vision Architectures
Under Distribution Shift and Long-tailed Data**](https://arxiv.org/pdf/2009.14193.pdf).

We include here code for evaluating conformal prediction results on distribution-shifted ImageNet datasets. To make this process more efficient, inference results (softmax values) from various models are pre-computed and saved.
The code for this is implemented in **save_results/**. For example, to save run inference using a ResNet-50 on ImageNet
and its variants,
run ``python save_results/save_results.py --exp_name resnet50 --model resnet50 -datasets IN1k INv2 INa INr``.

**main.py** runs conformal prediction + evaluation on using the inference results. The mean and variance of accuracy,
coverage, and inefficiency metrics are found across ```n_trials```. Currently,
the [THR / LABEL](https://arxiv.org/abs/1609.00451), [APS](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf),
and [RAPS](https://arxiv.org/pdf/2009.14193.pdf) conformal prediction methods are implemented. An example on how to
visualize results can be seen in the **plot_results** notebook.  

