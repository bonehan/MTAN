import torch
import torch.nn.functional as F
from torch import  nn
from config import args

class Conv3d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(channels_in,channels_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn = nn.BatchNorm3d(channels_out)
        self.activate = nn.ReLU(inplace = True)
    
    def forward(self, input):

        return self.activate(self.bn(self.conv(input)))#

class DCM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
        super(DCM, self).__init__()
        self.conv1 = Conv3d(channels_in,channels_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = Conv3d(channels_in + channels_out, channels_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv3 = Conv3d(channels_in + channels_out*2, channels_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.dropout1 = nn.Dropout3d(0.3)
    
    def forward(self, input):

        x1 = self.conv1(input)
        x2 = self.conv2(torch.cat([x1, input], 1))
        x3 = self.conv3(self.dropout1(torch.cat([x2, x1, input], 1)))

        return self.pool(x3)

class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.dcm1 = DCM(2 ,16, kernel_size = 3, stride = 1, padding = 1)
        self.dcm2 = DCM(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.dcm3 = DCM(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*27, 32)
        self.fc2 = nn.Linear(32, 8)
    
    def forward(self, input):

        x1 = self.dcm1(input)
        x2 = self.dcm2(x1)
        x3 = self.dcm3(x2)
        x3 = F.dropout(self.flatten(x3), 0.3)

        x4 = F.relu(self.fc1(x3))
        x5 = F.relu(self.fc2(x4))

        return x5

class wiseDNN(nn.Module):

    def __init__(self, num_landmarks = 40):
        super(wiseDNN, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.subnet = nn.ModuleDict()
        for i in range(self.num_landmarks):
            self.subnet['Landmark' + str(i)] = SubNet()
        
        self.fc1 = nn.Linear(8*self.num_landmarks, 128)
        self.fc2 = nn.Linear(128, args.score_num)

    def forward(self, inputs):

        local_embeddings = []
        for i in range(self.num_landmarks):
            local_embeddings.append(self.subnet['Landmark' + str(i)](inputs['landmark' + str(i)].cuda()))
        
        global_embeddings = torch.cat(local_embeddings, 1)
        x = F.relu(self.fc1(F.dropout(global_embeddings, 0.3)))

        return self.fc2(x)
