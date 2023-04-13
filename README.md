# Deep Metric Learning Research in PyTorch

---
## What can I find here?

This repository contains all code and implementations used in:

```
lm-metric: learned pair weighting and contextual memory for deep metric learning
```

**Link**: https://www.techrxiv.org/articles/preprint/LM-Metric_Learned_Pair_Weighting_and_Contextual_Memory_for_Deep_Metric_Learning/22361128/1/files/39799525.pdf

The code is based on ``https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch''
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

Otherwise, please make sure that the datasets have the following internal structure:

For CUB200-2011/CARS196:
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
For Stanford Online Products:
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
Assuming your folder is placed in e.g. <$datapath/cub200>, pass $datapath as input to --source.


Training:
Training is done by using main.py and setting the respective flags, all of which are listed and explained in parameters.py






