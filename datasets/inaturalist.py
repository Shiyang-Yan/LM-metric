import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, copy, torch, random, os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import scipy.io
from datasets.basic_dataset_scaffold import BaseDataset
flatten = lambda l: [item for sublist in l for item in sublist]
class TrainDatasetrsk(Dataset):
    def __init__(self, image_dict, opt):
        self.image_dict = image_dict
        self.dataset_name = opt.dataset
        self.batch_size = opt.bs
        self.samples_per_class = opt.samples_per_class
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        self.avail_classes = [*self.image_dict]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        transf_list.extend([
            transforms.RandomResizedCrop(size=224) if opt.arch in ['resnet50', 'resnet50_mcn', 'ViTB16','ViTB32','DeiTB'] else transforms.RandomResizedCrop(size=227),
            transforms.RandomHorizontalFlip(0.5)])
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)
        self.reshuffle()

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def reshuffle(self):
        image_dict = copy.deepcopy(self.image_dict)
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])
        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (
                        len(batch) < self.batch_size / self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:]
            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1
        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx):
        batch_item = self.dataset[idx]
        if self.dataset_name in ['Inaturalist']:
            cls = int(batch_item[0].split('/')[1])
        else:
            cls = batch_item[0]
        img = Image.open(batch_item[1])
        return cls, self.transform(self.ensure_3dim(img))

    def __len__(self):
        return len(self.dataset)


class BaseTripletDataset(Dataset):
    def __init__(self, image_dict, opt, samples_per_class=8, is_validation=False):
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])
        self.is_validation = is_validation
        self.pars        = opt
        self.image_dict  = image_dict
        self.avail_classes    = sorted(list(self.image_dict.keys()))
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([transforms.RandomResizedCrop(size=224) if opt.arch=='resnet50' or opt.arch=='ViTB16' or opt.arch=='ViTB32' or opt.arch=='DeiTB' else transforms.RandomResizedCrop(size=227),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256),
                                transforms.CenterCrop(224) if opt.arch=='resnet50' or opt.arch=='ViTB16' or opt.arch=='ViTB32' or opt.arch=='DeiTB' else transforms.CenterCrop(227)])
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]
        self.is_init = True

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.pars.loss == 'recallatk':
            if self.is_init:
                self.is_init = False
            if not self.is_validation:
                if self.samples_per_class==1:
                    return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
                if self.n_samples_drawn==self.samples_per_class:
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter: counter.remove(prev_class)
                    self.current_class   = counter[idx%len(counter)]
                    self.classes_visited = self.classes_visited+[self.current_class]
                    self.n_samples_drawn = 0
                class_sample_idx = idx%len(self.image_dict[self.current_class])
                self.n_samples_drawn += 1
                out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
                return self.current_class,out_img
            else:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
        else:
            if self.is_init:
                self.current_class = self.avail_classes[idx%len(self.avail_classes)]
                self.is_init = False
            if not self.is_validation:
                if self.samples_per_class==1:
                    return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
                if self.n_samples_drawn==self.samples_per_class:
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter: counter.remove(prev_class)
                    self.current_class   = counter[idx%len(counter)]
                    self.classes_visited = self.classes_visited[1:]+[self.current_class]
                    self.n_samples_drawn = 0
                class_sample_idx = idx%len(self.image_dict[self.current_class])
                self.n_samples_drawn += 1
                out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
                return self.current_class,out_img
            else:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files



def give_inaturalist_datasets(opt, datapath):
    train_image_dict, val_image_dict  = {},{}

    with open(os.path.join(datapath,'Inat_dataset_splits/Inaturalist_train_set1.txt')) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]
        for entry in FileLines:
            info = entry.split('/')
            if '/'.join([info[-3],info[-2]]) not in train_image_dict:
                train_image_dict['/'.join([info[-3],info[-2]])] = []
            train_image_dict['/'.join([info[-3],info[-2]])].append(os.path.join(datapath,entry))
    with open(os.path.join(datapath,'Inat_dataset_splits/Inaturalist_test_set1.txt')) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]
        for entry in FileLines:
            info = entry.split('/')
            if '/'.join([info[-3],info[-2]]) not in val_image_dict:
                val_image_dict['/'.join([info[-3],info[-2]])] = []
            val_image_dict['/'.join([info[-3],info[-2]])].append(os.path.join(datapath,entry))


    new_train_dict = {}
    class_ind_ind = 0
    for cate in train_image_dict:
        new_train_dict[class_ind_ind] = train_image_dict[cate]
        class_ind_ind += 1
    train_image_dict = new_train_dict


    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset = BaseTripletDataset(val_image_dict, opt, is_validation=True)
    eval_dataset = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset = BaseTripletDataset(train_image_dict, opt, is_validation=False)
    val_dataset = None

    return {'training': train_dataset, 'validation': val_dataset, 'testing': test_dataset, 'evaluation': eval_dataset, 'evaluation_train': eval_train_dataset}
