import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        ## Attention Maps 
        self.ATconvC1 = nn.Conv2d(3, 64, kernel_size=1)
        self.ATconvC2 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconvC3 = nn.Conv2d(3, 512, kernel_size=1)
        self.ATconvC4 = nn.Conv2d(3, 1024, kernel_size=1)
        self.ATconvC5 = nn.Conv2d(3, 2048, kernel_size=1)

        self.ATconv3 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconv4 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconv5 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconv6 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconv7 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconv8 = nn.Conv2d(3, 256, kernel_size=1)
        self.ATconv9 = nn.Conv2d(3, 256, kernel_size=1)

        ### Attention Downsample Layers
        self.downsampleC1 = nn.Upsample(size=(128, 128), mode='bilinear')
        self.downsampleC2 = nn.Upsample(size=(128, 128), mode='bilinear')
        self.downsampleC3 = nn.Upsample(size=(64, 64), mode='bilinear')
        self.downsampleC4 = nn.Upsample(size=(32, 32), mode='bilinear')
        self.downsampleC5 = nn.Upsample(size=(16, 16), mode='bilinear')

        self.downsample3 = nn.Upsample(size=(64, 64), mode='bilinear')
        self.downsample4 = nn.Upsample(size=(32, 32), mode='bilinear')
        self.downsample5 = nn.Upsample(size=(16, 16), mode='bilinear')
        self.downsample6 = nn.Upsample(size=(8, 8), mode='bilinear')
        self.downsample7 = nn.Upsample(size=(4, 4), mode='bilinear')
        self.downsample8 = nn.Upsample(size=(2, 2), mode='bilinear')
        self.downsample9 = nn.Upsample(size=(1, 1), mode='bilinear')



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x, att_map):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        atC1 = self.ATconvC1(self.downsampleC1(att_map))
        c1 = c1 + atC1

        c2 = self.layer1(c1)
        atC2 = self.ATconvC2(self.downsampleC2(att_map))
        c2 = c2 + atC2
        
        c3 = self.layer2(c2)
        atC3 = self.ATconvC3(self.downsampleC3(att_map))
        c3 = c3 + atC3
        
        c4 = self.layer3(c3)
        atC4 = self.ATconvC4(self.downsampleC4(att_map))
        c4 = c4 + atC4
        
        c5 = self.layer4(c4)
        atC5 = self.ATconvC5(self.downsampleC5(att_map))
        c5 = c5 + atC5
        
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p8 = self.conv8(F.relu(p7))
        p9 = self.conv9(F.relu(p8))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
     

        # Attention Computation
        at3 = self.ATconv3(self.downsample3(att_map))
        at4 = self.ATconv4(self.downsample4(att_map))
        at5 = self.ATconv5(self.downsample5(att_map))
        at6 = self.ATconv6(self.downsample6(att_map))
        at7 = self.ATconv7(self.downsample7(att_map))
        at8 = self.ATconv8(self.downsample8(att_map))
        at9 = self.ATconv9(self.downsample9(att_map))

        return p3 + at3, p4 + at4, p5 + at5, p6 + at6, p7 + at7, p8 + at8, p9 + at9


def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [3,4,23,3])

def FPN152():
    return FPN(Bottleneck, [3,8,36,3])


def test():
    net = FPN50()
    fms = net(torch.randn(1,3,512,512))
    for fm in fms:
        print(fm.size())

# test()
