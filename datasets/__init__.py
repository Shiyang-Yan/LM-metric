import datasets.cub200
import datasets.cars196
import datasets.stanford_online_products
import datasets.inaturalist
from datasets.sop import *
def select(dataset, opt, data_path, mode='train'):
    if 'cub200' in dataset:
        return cub200.Give(opt, data_path)

    if 'cars196' in dataset:
        return cars196.Give(opt, data_path)
        
    if 'sop' in dataset:
        return SOPDataset(data_path, mode=mode)

    if 'online_products' in dataset:
        return stanford_online_products.Give(opt, data_path)

    if 'inaturalist' in dataset:
        return inaturalist.give_inaturalist_datasets(opt, data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196 & online_products!'.format(dataset))
