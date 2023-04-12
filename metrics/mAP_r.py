import torch
import numpy as np
import faiss



class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features', 'target_labels']
        self.name     = 'mAP_r'

    def __call__(self, target_labels, features):
        from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
        target_labels = torch.from_numpy(target_labels).cuda()
        features = torch.from_numpy(features).cuda()
        calculator = AccuracyCalculator(device=None, k = 2047)

        acc_dict = calculator.get_accuracy(features,
                                           features,
                                           target_labels.squeeze(),
                                           target_labels.squeeze(),
                                           embeddings_come_from_same_source=True
                                           )
        print (acc_dict)

        return 0
