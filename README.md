# Deep Metric Learning Research in PyTorch

---
## What can I find here?

This repository contains all code and implementations used in:

```
lm-metric: learned pair weighting and contextual memory for deep metric learning
```

``Abstract: In many real-world applications involving sample alignment and matching, such as classical image retrieval, it is challenging to optimize a ranking-based metric, e.g., average precision (AP), as it involves a discrete ranking function that is not only non-differentiable but also non-decomposable. “Non-differentiable” means that AP is not trainable via backpropagation. “Non-decomposable” means that AP is not linearly composable via the linear combination of batches for a dataset. In this study, we approach this problem from two novel perspectives: optimal pair weighting and inter-sample relationship. First, we introduce a parametric pairwise weighting scheme via policy gradient optimization and model the batch-wise inter-sample relationship via a Gated Recurrent Unit (GRU). We propose a conditional Normalizing Flow-based contextual memory feature block to learn a compact single embedding for each image containing the contextual information during retrieval. Our method is comprehensively evaluated, and significant improvements have been achieved in large-scale image retrieval benchmark datasets. We call our method LM-Metric (Learned Pair Weighting and Contextual Memory for Deep Metric Learning). '' 

**Link**: https://www.techrxiv.org/articles/preprint/LM-Metric_Learned_Pair_Weighting_and_Contextual_Memory_for_Deep_Metric_Learning/22361128/1/files/39799525.pdf

The code is based on ``https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch''.
Data preparation is the same to that repo.


**Contact**: elyotyan@gmail.com

*Suggestions are always welcome!*

Requirements:
PyTorch 1.2.0+ & Faiss-Gpu
Python 3.6+
pretrainedmodels, torchvision 0.3.0+


Datasets:
Data for

CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)

CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)

can be downloaded either from the respective project sites or directly via Dropbox:

CUB200-2011 (1.08 GB): https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0

CARS196 (1.86 GB): https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0

SOP (2.84 GB): https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0

The latter ensures that the folder structure is already consistent with this pipeline and the dataloaders.


Training:
Training is done by using main.py and setting the respective flags, all of which are listed and explained in parameters.py






