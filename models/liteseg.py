import torch 

# from models import model_shufflenet_modified as shufflenet
from models import liteseg_mobilenet as mobilenet

from models import liteseg_shufflenet as shufflenet

class LiteSeg():
    
        
    def build(backbone_network,modelpath,CONFIG,is_train=True):
                
        if backbone_network.lower() == 'shufflenet':
            net = shufflenet.RT(n_classes=5, pretrained=is_train, PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_SHUFFLENET)
        elif backbone_network.lower() == 'mobilenet':
            net = mobilenet.RT(n_classes=5,pretrained=is_train, PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_MOBILENET)
        else:
            raise NotImplementedError
            
        if modelpath is not None:
            net.load_state_dict(torch.load(modelpath))
            
        print("Using LiteSeg with",backbone_network)
        return net