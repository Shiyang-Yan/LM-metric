"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import faiss
import parameters    as par

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)

##### Read in parameters
opt = parser.parse_args()


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
if opt.savename=='group_plus_seed':
    if opt.log_online:
        opt.savename = opt.group+'_s{}'.format(opt.seed)
    else:
        opt.savename = ''

### If wandb-logging is turned on, initialize the wandb-run here:
if opt.log_online:
    import wandb
    _ = os.system('wandb login {}'.format(opt.wandb_key))
    os.environ['WANDB_API_KEY'] = opt.wandb_key
    wandb.init(project=opt.project, group=opt.group, name=opt.savename, dir=opt.save_path)
    wandb.config.update(opt)



"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import metrics       as metrics
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger
torch.backends.cudnn.enabled = False


"""==================================================================================================="""
full_training_start_time = time.time()



"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained = not opt.not_pretrained




"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
# if not opt.use_data_parallel:
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])



"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)



"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
model = archs.select(opt.arch, opt)

opt.device = torch.device('cuda')
model = archs.select(opt.arch, opt)
model.load_state_dict(torch.load('/mnt/disk2/sxy/fewshot/code/sop_smoothap/sop/checkpoint_Test_discriminative_e_recall@1.pth.tar')['state_dict'], strict=False)

_  = model.to(opt.device)


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def inner_product(qf, gf):
    return (1. - qf.mm(gf.t()))


def ap_c(target_labels, features_cosine):
        labels, freqs = np.unique(target_labels, return_counts=True)
        R             = np.max(freqs)

        faiss_search_index  = faiss.IndexFlatIP(features_cosine.shape[-1])
        if isinstance(features_cosine, torch.Tensor):
            features_cosine = features_cosine.detach().cpu().numpy()
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
        faiss_search_index.add(features_cosine)
        nearest_neighbours  = faiss_search_index.search(features_cosine, int(R+1))[1][:,1:]

        target_labels = target_labels.reshape(-1)
        nn_labels = target_labels[nearest_neighbours]

        avg_r_precisions = []
        for label, freq in zip(labels, freqs):
            rows_with_label = np.where(target_labels==label)[0]
            for row in rows_with_label:
                n_recalled_samples           = np.arange(1,freq+1)
                target_label_occ_in_row      = nn_labels[row,:freq]==label
                cumsum_target_label_freq_row = np.cumsum(target_label_occ_in_row)
                avg_r_pr_row = np.sum(cumsum_target_label_freq_row*target_label_occ_in_row/n_recalled_samples)/freq
                avg_r_precisions.append(avg_r_pr_row)

        return (avg_r_precisions)

def AP_evaluation(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    q_camids = np.ones(q_pids.shape[0])
    g_camids = np.zeros(q_pids.shape[0])
    if num_g < max_rank:
        max_rank = num_g
        #print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []

    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(torch.tensor(cmc[:max_rank]).sum())
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum() + 1e-12
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    return torch.tensor(all_AP)


"""============================================================================"""
#################### DATALOADER SETUPS ##################
if __name__ == '__main__':
    dataloaders = {}
    datasets_train    = datasets.select(opt.dataset, opt, opt.source_path)
    datasets_test     = datasets.select(opt.dataset, opt, opt.source_path, mode='test')
    dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets_test, num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
    dataloaders['testing']    = torch.utils.data.DataLoader(datasets_test,  num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
    if opt.use_tv_split:
        dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels, batch_size=opt.bs,shuffle=False)

    train_data_sampler      = dsamplers.select(opt.data_sampler, opt, datasets_train)
    #if train_data_sampler.requires_storage:
    #    train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

    dataloaders['training'] = torch.utils.data.DataLoader(datasets_train, num_workers=opt.kernels, batch_sampler=train_data_sampler)

    #opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)




    """============================================================================"""
    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Test', 'Model Grad']
    if opt.use_tv_split: sub_loggers.append('Val')
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)





    """============================================================================"""
    #################### LOSS SETUP ####################


    batchminer   = bmine.select(opt.batch_mining, opt)
    if opt.fc_lr<0:
        to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
    else:
        all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'mapping' not in x[0], model.named_parameters()))]
        fc_params         = model.model.mapping.parameters()
        to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.fc_lr,'weight_decay':opt.decay}]
    criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
    _ = criterion.to(opt.device)

    #if 'criterion' in train_data_sampler.name:
    #    train_data_sampler.internal_criterion = criterion




    """============================================================================"""
    #################### OPTIM SETUP ####################
    if opt.optim == 'adam':
        optimizer    = torch.optim.Adam(to_optim)
    elif opt.optim == 'sgd':
        optimizer    = torch.optim.SGD(to_optim, momentum=0.9)
    elif opt.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(to_optim)
    else:
        raise Exception('Optimizer <{}> not available!'.format(opt.optim))



    #lr_decay_step  = 10
    #lr_decay_gamma = 0.5
    ##scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma = lr_decay_gamma)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)


    """============================================================================"""
    #################### METRIC COMPUTER ####################
    opt.rho_spectrum_embed_dim = opt.embed_dim
    metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)





    """============================================================================"""
    ################### Summary #########################3
    data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
    setup_text = 'Objective:\t {}'.format(opt.loss.upper())
    miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
    arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
    summary    = data_text+'\n'+setup_text+'\n'+miner_text+'\n'+arch_text
    print(summary)




    """============================================================================"""
    ################### SCRIPT MAIN ##########################
    print('\n-----\n')

    iter_count = 0
    loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}

    for epoch in range(1):
        epoch_start_time = time.time()

        if epoch>0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
            train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

        opt.epoch = epoch
        ### Scheduling Changes specifically for cosine scheduling
        if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

        """======================================="""
        #if train_data_sampler.requires_storage:
        #    train_data_sampler.precompute_indices()


        """======================================="""
        ### Train one epoch
        start = time.time()
        _ = model.train()


        loss_collect = []
        data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))




        ### Evaluate Metric for Training & Test (& Validation)
        if epoch % 1 == 0:
            _ = model.eval()
            print('\nComputing Testing Metrics...')
            eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing']],    model, opt, opt.evaltypes, opt.device, log_key='Test')
            if opt.use_tv_split:
                print('\nComputing Validation Metrics...')
                eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['validation']], model, opt, opt.evaltypes, opt.device, log_key='Val')
           # print('\nComputing Training Metrics...')
           # eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['evaluation']], model, opt, opt.evaltypes, opt.device, log_key='Train')


            LOG.update(all=True)


        """======================================="""
         ### Learning Rate Scheduling Step
        if opt.scheduler != 'none':
            scheduler.step()

        print('Total Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
        print('\n-----\n')




    """======================================================="""
    ### CREATE A SUMMARY TEXT FILE
    summary_text = ''
    full_training_time = time.time()-full_training_start_time
    summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))

    summary_text += '---------------\n'
    for sub_logger in LOG.sub_loggers:
        metrics       = LOG.graph_writer[sub_logger].ov_title
        summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), metrics)

    with open(opt.save_path+'/training_summary.txt','w') as summary_file:
        summary_file.write(summary_text)
