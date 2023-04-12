"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm
import torch.nn.functional as F

from pytorch_revgrad import RevGrad
import torch.nn.init as init
from models import Glow
import util

def sample(net, gray_img, sigma=0.6):
    B, C, W, H = gray_img.shape
    z = torch.randn((B, 1, 16, 8), dtype=torch.float32, device='cuda') * sigma
    x, _ = net(z, gray_img, reverse=True)
    x = torch.sigmoid(x)

    return x
"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.opt = opt
        beta = 64
        self.embed_size = beta * 2
        self.aspace = torch.tensor(range(1, 2 * beta + 1)).cuda().float() / (2*beta)
        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else None)
        self.model.smachine = nn.GRUCell(self.model.last_linear.in_features, self.model.last_linear.in_features)
        self.model.decision_fcsp = nn.Linear(self.model.last_linear.in_features, self.embed_size)
        self.model.decision_fcsn = nn.Linear(self.model.last_linear.in_features, self.embed_size)
        self.name = opt.arch
        self.model.transform = nn.Linear(opt.embed_dim, opt.embed_dim, bias=False)
        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.mapping = nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.loss_fn = util.NLLLoss().to('cuda')

        self.model.out_adjust = None
        self.model.gflow = Glow(num_channels=128,
               num_levels=2,
               num_steps=8,
               mode="sketch")
        self.weight_cliping_limit = 0.01
    def forward(self, x,  meta_train, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x
        avg_x = self.model.gap(no_avg_feat)
        max_x = self.model.gmp(no_avg_feat)
        x = avg_x# + max_x
        enc_out = x = x.view(x.size(0), -1)
        state = torch.zeros(1, self.model.last_linear.in_features).cuda()
        action_all_p = []
        action_all_n = []
        probp = []
        probn = []
        outputs = []
        for i in range(x.size(0)):
            state = self.model.smachine(x[i, :].unsqueeze(0), state)
            outputs.append(state)
            outputp = (self.model.decision_fcsp(F.relu(state + x[i, :].unsqueeze(0))))
            outputp = F.softmax(outputp, -1)
            outputn = (self.model.decision_fcsn(F.relu(state + x[i, :].unsqueeze(0))))
            outputn = F.softmax(outputn, -1)
            distp = torch.distributions.Categorical(torch.nan_to_num(outputp))
            actionp = distp.sample()
            probp.append(distp.log_prob(actionp))
            distn = torch.distributions.Categorical(torch.nan_to_num(outputn))
            actionn = distn.sample()
            probn.append(distn.log_prob(actionn))
            action_utilized_p = self.aspace[actionp]
            action_utilized_n = self.aspace[actionn]
            action_all_p.append(action_utilized_p)
            action_all_n.append(action_utilized_n)
        action_all_p = torch.stack(action_all_p).squeeze()
        action_all_n = torch.stack(action_all_n).squeeze()
        probp = torch.stack(probp).squeeze()
        probn = torch.stack(probn).squeeze()
        outputs = torch.stack(outputs).squeeze()
        rnn_output = F.sigmoid(self.model.mapping(outputs.squeeze()))
        embedding_1st = self.model.mapping(x.squeeze())
        ##########
        z, sldj = self.model.gflow(rnn_output.detach().view(-1, 1, 16, 8), embedding_1st.view(-1, 1, 16, 8).detach(), reverse=False)
        loss = self.loss_fn(z, sldj)/100.

        if self.model.training==True:
            embedding_2nd = rnn_output
        else:
            embedding_2nd = sample(self.model.gflow, embedding_1st.view(-1, 1, 16, 8))

        embedding = embedding_1st + embedding_2nd.view(-1, 512)

        if 'normalize' in self.pars.arch:
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            rnn_output = torch.nn.functional.normalize(rnn_output, dim=-1)
        if self.model.out_adjust and not self.train:
            embedding = self.model.out_adjust(embedding)

        return (embedding, rnn_output, (enc_out, no_avg_feat)), (action_all_p, action_all_n), (probp, probn), loss

    #def _initialize_weights(self):
    ##    init.kaiming_normal_(self.model.last_linear_embedding.weight, mode='fan_out')
    #     init.constant_(self.model.last_linear_embedding.bias, 0)