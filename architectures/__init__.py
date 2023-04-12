import architectures.resnet50
import architectures.googlenet
import architectures.bninception
import architectures.vit
def select(arch, opt):
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
    if 'vit'in arch:
        return vit.Network(opt)
