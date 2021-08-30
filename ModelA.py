import torch
import torch.nn as nn 
from Layers import  DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm, fc_layer
import copy
import os.path
import torchvision
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_length=1024):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.embedP = Embedder(max_length, d_model)
       # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self,feature,step ,trg_mask):
        position = torch.arange(0, feature.size(1), dtype=torch.long,
                                    device=feature.device)

        x = feature+self.embedP(position)+self.embed(step)*0

        for i in range(self.N):
            x = self.layers[i](x,  trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self,  trg_vocab, d_model, N, heads, dropout,feature_shape=6*6*2048):
        super().__init__()
        self.feature= fc_layer(feature_shape,d_model)



        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = fc_layer(d_model, trg_vocab)

    def forward(self, feature ,step, trg_mask):
        feature=self.feature(feature)

        d_output = self.decoder(feature,step , trg_mask)
        output = self.out(d_output)
        return output

class RESNET_Transformer(nn.Module):
    def __init__(self,  trg_vocab, d_model, N, heads, dropout,feature_shape=1000):
        super().__init__()
        self.feature= fc_layer(feature_shape,d_model)

        self.resnet = torchvision.models.resnet18(pretrained=False).eval().requires_grad_(True)

        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = fc_layer(d_model, trg_vocab)

    def forward(self, feature , trg_mask):
        x=self.resnet(feature).unsqueeze(0)
        feature=self.feature(x)

        d_output = self.decoder(feature,  trg_mask)
        output = self.out(d_output)
        output=output[:,-1,:]
        return output


def get_model(opt,  trg_vocab,model_weights='model_weights'):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer( trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None and os.path.isfile(opt.load_weights+'/'+model_weights):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/'+model_weights))
    else:
        total_param = 0
        for p in model.parameters():
            if p.dim() > 1:
                #nn.init.xavier_uniform_(p)
                a=0
            length = len(p.shape)
            param = 1
            for j in range(length):
                param = p.shape[j] * param

            total_param += param
        print('使用参score:{}百万'.format(total_param/1000000))
    return model


def get_modelB(opt, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = RESNET_Transformer(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None and os.path.isfile(opt.load_weights + '/model_weightsB'):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weightsB'))
    else:
        total_param = 0
        for p in model.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                a = 0
            length = len(p.shape)
            param = 1
            for j in range(length):
                param = p.shape[j] * param

            total_param += param
        print('使用参score:{}百万'.format(total_param / 1000000))
    return model