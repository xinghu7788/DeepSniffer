import torch
import torch.nn as nn
import torch.nn.functional as F

inf = {
    'c0' : [3, 64, 7, 3, 2],
    'b00' : [64, 64, 3, 1, 1], 'b01' : [64, 64, 3, 1, 1], 'b10' : [64, 64, 3, 1, 1], 'b11' : [64, 64, 3, 1, 1],
    'b20' : [64, 128, 3, 1, 2], 'b21' : [128, 128, 3, 1, 1], 'b30' : [128, 128, 3, 1, 1], 'b31' : [128, 128, 3, 1, 1],
    'b40' : [128, 256, 3 ,1, 2], 'b41' : [256, 256, 3, 1, 1], 'b50' : [256, 256, 3, 1, 1], 'b51' : [256, 256, 3, 1, 1],
    'b60' : [256, 512, 3, 1, 2], 'b61' : [512, 512, 3, 1, 1], 'b70' : [512, 512, 3, 1, 1], 'b71' : [512, 512, 3, 1, 1],
    'cut0' : [64, 128, 1, 0, 2], 'cut1' : [128, 256, 1, 0, 2], 'cut2' : [256, 512, 1, 0, 2],
    'avg' : [1]
}

def conv(cur_inf):
    return nn.Conv2d(cur_inf[0], cur_inf[1], kernel_size=cur_inf[2], stride=cur_inf[4], padding=cur_inf[3], bias=False)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        global inf



        self.c0 = conv(inf['c0'])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b00 = conv(inf['b00'])
        self.bn00 = nn.BatchNorm2d(inf['b00'][1])
        self.b01 = conv(inf['b01'])
        self.bn01 = nn.BatchNorm2d(inf['b01'][1])

        self.b10 = conv(inf['b10'])
        self.bn10 = nn.BatchNorm2d(inf['b10'][1])
        self.b11 = conv(inf['b11'])
        self.bn11 = nn.BatchNorm2d(inf['b11'][1])

        self.b20 = conv(inf['b20'])
        self.bn20 = nn.BatchNorm2d(inf['b20'][1])
        self.b21 = conv(inf['b21'])
        self.bn21 = nn.BatchNorm2d(inf['b21'][1])
        self.cut0 = conv(inf['cut0'])
        self.bncut0 = nn.BatchNorm2d(inf['cut0'][1])

        self.b30 = conv(inf['b30'])
        self.bn30 = nn.BatchNorm2d(inf['b30'][1])
        self.b31 = conv(inf['b31'])
        self.bn31 = nn.BatchNorm2d(inf['b31'][1])

        self.b40 = conv(inf['b40'])
        self.bn40 = nn.BatchNorm2d(inf['b40'][1])
        self.b41 = conv(inf['b41'])
        self.bn41 = nn.BatchNorm2d(inf['b41'][1])
        self.cut1 = conv(inf['cut1'])
        self.bncut1 = nn.BatchNorm2d(inf['cut1'][1])

        self.b50 = conv(inf['b50'])
        self.bn50 = nn.BatchNorm2d(inf['b50'][1])
        self.b51 = conv(inf['b51'])
        self.bn51 = nn.BatchNorm2d(inf['b51'][1])

        self.b60 = conv(inf['b60'])
        self.bn60 = nn.BatchNorm2d(inf['b60'][1])
        self.b61 = conv(inf['b61'])
        self.bn61 = nn.BatchNorm2d(inf['b61'][1])
        self.cut2 = conv(inf['cut2'])
        self.bncut2 = nn.BatchNorm2d(inf['cut2'][1])

        self.b70 = conv(inf['b70'])
        self.bn70 = nn.BatchNorm2d(inf['b70'][1])
        self.b71 = conv(inf['b71'])
        self.bn71 = nn.BatchNorm2d(inf['b71'][1])

        self.avgpool = nn.AvgPool2d(7, stride=inf['avg'][0])
        self.fc = nn.Linear(512, 1000)


    def forward(self, x):

        x = F.relu(self.c0(x))
        x = self.maxpool(x)

        tmp = F.relu(self.bn00(self.b00(x)))
        tmp = self.bn01(self.b01(tmp))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn10(self.b10(x)))
        tmp = self.bn11(self.b11(tmp))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn20(self.b20(x)))
        tmp = self.bn21(self.b21(tmp))
        x = self.bncut0(self.cut0(x))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn30(self.b30(x)))
        tmp = self.bn31(self.b31(tmp))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn40(self.b40(x)))
        tmp = self.bn41(self.b41(tmp))
        x = self.bncut1(self.cut1(x))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn50(self.b50(x)))
        tmp = self.bn51(self.b51(tmp))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn60(self.b60(x)))
        tmp = self.bn61(self.b61(tmp))
        x = self.bncut2(self.cut2(x))
        x = x + tmp
        x = F.relu(x)

        tmp = F.relu(self.bn70(self.b70(x)))
        tmp = F.relu(self.bn71(self.b71(tmp)))
        x = x + tmp
        x = F.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def fake_4(pretrained):
    global inf
    inf = {
        'c0': [3, 64, 5, 2, 2],
        'b00': [64, 64, 5, 2, 1], 'b01': [64, 64, 5, 2, 1], 'b10': [64, 64, 5, 2, 1], 'b11': [64, 64, 5, 2, 1],
        'b20': [64, 128, 5, 2, 2], 'b21': [128, 128, 5, 2, 1], 'b30': [128, 128, 5, 2, 1], 'b31': [128, 128, 5, 2, 1],
        'b40': [128, 256, 3, 1, 2], 'b41': [256, 256, 3, 1, 1], 'b50': [256, 256, 3, 1, 1], 'b51': [256, 256, 3, 1, 1],
        'b60': [256, 512, 3, 1, 2], 'b61': [512, 512, 3, 1, 1], 'b70': [512, 512, 3, 1, 1], 'b71': [512, 512, 3, 1, 1],
        'cut0': [64, 128, 1, 0, 2], 'cut1': [128, 256, 1, 0, 2], 'cut2': [256, 512, 1, 0, 2],
        'avg': [1]
    }
    model = ResNet()
    if pretrained != 'null':
        model.load_state_dict(torch.load(pretrained))
    return model


def fake_5(pretrained):
    global inf
    inf = {
        'c0': [3, 64, 7, 3, 2],
        'b00': [64, 64, 3, 1, 1], 'b01': [64, 64, 3, 1, 1], 'b10': [64, 64, 3, 1, 1], 'b11': [64, 64, 3, 1, 1],
        'b20': [64, 128, 3, 1, 2], 'b21': [128, 128, 3, 1, 1], 'b30': [128, 128, 3, 1, 1], 'b31': [128, 128, 3, 1, 1],
        'b40': [128, 256, 5, 2, 2], 'b41': [256, 256, 5, 2, 1], 'b50': [256, 256, 5, 2, 1], 'b51': [256, 256, 5, 2, 1],
        'b60': [256, 512, 5, 2, 2], 'b61': [512, 512, 5, 2, 1], 'b70': [512, 512, 5, 2, 1], 'b71': [512, 512, 5, 2, 1],
        'cut0': [64, 128, 1, 0, 2], 'cut1': [128, 256, 1, 0, 2], 'cut2': [256, 512, 1, 0, 2],
        'avg': [1]
    }
    model = ResNet()
    if pretrained != 'null':
        model.load_state_dict(torch.load(pretrained))
    return model













