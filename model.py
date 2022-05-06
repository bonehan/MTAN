import torch
import torch.nn.functional as F
from einops import repeat
from torch import Tensor, nn

from config import args

class Conv3d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(channels_in,channels_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn = nn.BatchNorm3d(channels_out)
        self.activate = nn.ReLU(inplace = True)
    
    def forward(self, input):

        return self.activate(self.bn(self.conv(input)))#

class MultiheadAttention(nn.Module):

    def __init__(self, model_dim=64, num_heads=4, dropout=0.3):
        super(MultiheadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size, src_len, _ = key.size()
        tgt_len = query.size(1)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5

        attention = torch.bmm(query, key.transpose(1, 2))*scale
        context = torch.bmm(self.dropout(self.softmax(attention)), value)

        # print(attention.shape)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # linear projection
        output = self.linear(self.dropout(context))
        attention = attention.view(batch_size, num_heads, tgt_len, src_len)
        attention = torch.sigmoid(attention).sum(1)/num_heads

        return output, attention

class HieAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(HieAttentionLayer, self).__init__()
        self.Projection = nn.Sequential(
            Conv3d(args.Filters[5], args.Filters[6], kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[6], args.Filters[7], kernel_size = 3, stride = 1, padding = 1),
            nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(args.dropout),
            nn.Linear(args.Filters[7], args.Latent),
            nn.ReLU(inplace = True),            
        )
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        
        b, c, h, w, d = tgt.size()
        tgt1 = self.Projection(tgt).unsqueeze(1)
        memory = memory.transpose(1, 0)

        tgt2, att = self.multihead_attn(tgt1.detach(), memory, memory)
        
        Attn = att.view((b, 1, h, w, d))
        tgt3 = self.Projection(tgt * Attn.detach())

        return tgt1.squeeze(1), tgt2.squeeze(1), tgt3, Attn
        
class VIT(nn.Module):

    def __init__(self, pool = 'cls'):
        super(VIT, self).__init__()

        self.encoder = Encoder()

        self.prediction = nn.ModuleDict({
            'MMSE': Prediction('none'),
            'CDRSB': Prediction('none'),
            'ADAS11': Prediction('none'),
            'ADAS13': Prediction('none'),
        })

        num_patches = 9*12*9
        self.pos_embedding = nn.Parameter(torch.randn(num_patches+1, 1, args.Filters[5]))
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.Filters[5]))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(args.dropout)
        self.heads = 4
        self.tranformerlayer = nn.TransformerEncoderLayer(args.Filters[5], self.heads, dim_feedforward=self.heads*16, dropout=args.dropout)
        self.transformer = nn.TransformerEncoder(self.tranformerlayer, 2)

        self.pool = pool 

    def forward(self, inputs):
        
        x = self.encoder(inputs)
        # print(x.shape)
        b, c, h, w, d = x.shape 
        x = x.view((b, c, h*w*d)).contiguous().transpose(2, 1)
        x = x.transpose(1, 0)
        # print(x.shape)      

        cls_tokens = repeat(self.cls_token, 'n () d -> n b d', b=b) 
        # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=0)            
        x += self.pos_embedding[:, :(h*w*d+1)]                 
        x = self.dropout(x)

        x = self.transformer(x).transpose(0, 1)  
        # print(x.shape)                                             

        x = x[:, 1:].mean(dim=1) if self.pool == 'mean' else x[:, 0]                                                   
        # print(x.shape)

        cdr, adas11, adas13, mmse = self.prediction['CDRSB'](x), self.prediction['ADAS11'](x), self.prediction['ADAS13'](x), self.prediction['MMSE'](x)

        return cdr, adas11, adas13, mmse

class Prediction(nn.Module):
    def __init__(self, mode = 'concat'):
        super(Prediction, self).__init__()
        self.mode = mode

        if self.mode =='concat':
            self.fc1 = nn.Linear(args.Filters[4]*3, 16)
        else:
            self.fc1 = nn.Linear(args.Filters[4], 16)

        self.fc2 = nn.Linear(16, 1)

    def forward(self, inputs):

        return self.fc2(F.relu(self.fc1(inputs)))

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            Conv3d(1, args.Filters[0], kernel_size = 3, stride = 1, padding = 1),
            Conv3d(args.Filters[0], args.Filters[1], kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool3d((2, 2, 2)),
            # nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[1], args.Filters[2], kernel_size = 3, stride = 1, padding = 1),
            Conv3d(args.Filters[2], args.Filters[3], kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool3d((2, 2, 2)),
            # nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[3], args.Filters[4], kernel_size = 3, stride = 1, padding = 1),
            Conv3d(args.Filters[4], args.Filters[4], kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool3d((2, 2, 2)),
            # nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[4], args.Filters[4], kernel_size = 3, stride = 1, padding = 1),
            Conv3d(args.Filters[4], args.Filters[4], kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(args.dropout),
        )

    def forward(self, inputs):

        return self.encoder(inputs)

class MTAN(nn.Module):

    def __init__(self, mode = 'concat'):
        super(MTAN, self).__init__()

        self.encoder = Encoder()

        num_patches = 9*12*9
        self.pos_embedding = nn.parameter.Parameter(torch.randn(num_patches, 1, args.Filters[4]))
        self.dropout = nn.Dropout(args.dropout)
        self.heads = 4
        self.tranformerlayer = nn.TransformerEncoderLayer(args.Filters[4], self.heads, dim_feedforward=self.heads*16, dropout=args.dropout)
        self.transformer = nn.TransformerEncoder(self.tranformerlayer, 2)
        self.attn = HieAttentionLayer(args.Filters[4], self.heads, dropout=args.dropout)
        self.mode = mode
        self.prediction = nn.ModuleDict({
            'MMSE': Prediction(self.mode),
            'CDRSB': Prediction(self.mode),
            'ADAS11': Prediction(self.mode),
            'ADAS13': Prediction(self.mode),
        })
    
    def forward(self, inputs):
        
        Embedding = self.encoder(inputs)
        # print(Embedding.shape)
        b, c, h, w, d = Embedding.shape 
        x = Embedding.view((b, c, h*w*d)).contiguous().transpose(2, 1)
        x = x.transpose(1, 0)     
        x += self.pos_embedding              
        x = self.dropout(x)
        Tokens = self.transformer(x)  

        tgt1, tgt2, tgt3, Attn = self.attn(Embedding, Tokens)

        if self.mode == 'mean':
            Fuse = torch.mean(torch.cat([tgt1.unsqueeze(1), tgt2.unsqueeze(1), tgt3.unsqueeze(1)], 1), 1)
        if self.mode == 'max':
            Fuse, _ = torch.max(torch.cat([tgt1.unsqueeze(1), tgt2.unsqueeze(1), tgt3.unsqueeze(1)], 1), 1)
        if self.mode == 'concat':
            Fuse = torch.cat([tgt1, tgt2, tgt3], -1)

        cdr, adas11, adas13, mmse = self.prediction['CDRSB'](Fuse), self.prediction['ADAS11'](Fuse), self.prediction['ADAS13'](Fuse), self.prediction['MMSE'](Fuse) 
        
        return cdr, adas11, adas13, mmse, Attn

class CNN_M(nn.Module):
    def __init__(self):
        super(CNN_M, self).__init__()

        self.encoder = Encoder()
        self.Projection = nn.Sequential(
            Conv3d(args.Filters[5], args.Filters[6], kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[6], args.Filters[7], kernel_size = 3, stride = 1, padding = 1),
            nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(args.dropout),
            nn.Linear(args.Filters[7], args.Latent),
            nn.ReLU(inplace = True),            
        )
        self.prediction = nn.ModuleDict({
            'MMSE': Prediction('none'),
            'CDRSB': Prediction('none'),
            'ADAS11': Prediction('none'),
            'ADAS13': Prediction('none'),
        })


    def forward(self, inputs):

        x = self.encoder(inputs)
        x = self.Projection(x)
        cdr, adas11, adas13, mmse = self.prediction['CDRSB'](x), self.prediction['ADAS11'](x), self.prediction['ADAS13'](x), self.prediction['MMSE'](x)

        return cdr, adas11, adas13, mmse

