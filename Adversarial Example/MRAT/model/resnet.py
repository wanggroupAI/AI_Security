import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

ngf = 128


    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        x2 = F.relu(self.bn1(self.conv1(x)))
        x3 = self.bn2(self.conv2(x2))
        x4 = x3 + self.shortcut(x)
        x5 = F.relu(x4)
        return x5


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        x2 = F.relu(self.bn1(self.conv1(x)))
        x3 = F.relu(self.bn2(self.conv2(x2)))
        x4 = self.bn3(self.conv3(x3))
        x5 = x4 + self.shortcut(x)
        x6 = F.relu(x5)
        return x6
    
    
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == 'replicate':
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block.extend([nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)])
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        p = 0
        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == 'replicate':
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block.extend([nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)])

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, img_size=(32,32)):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.is_mask = False

        decoder_lis = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(size=img_size, mode="bilinear", align_corners=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        ]     



        bottle_neck_lis = [
                            ResnetBlock(512),
                            ResnetBlock(512),
                            ResnetBlock(512),
                            ResnetBlock(512),
                           ]

        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.maxpool(self.relu(self.bn(self.conv(x))))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = torch.flatten(x6, 1)
        out = self.fc(x7)
        
        if self.is_mask:
            reimage = self.bottle_neck(x5)
            reimage2 = self.decoder(reimage)                
            return out, reimage2
        else:
            return out

    
    def turn_on_mask(self):
        self.is_mask = True
        
    def turn_off_mask(self):
        self.is_mask = False



def resnet18(num_classes=10, img_size=(32,32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, img_size=img_size)


def resnet34(num_classes=10, img_size=(32,32)):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, img_size=img_size)


def resnet50(num_classes=10, img_size=(32,32)):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, img_size=img_size)


def resnet101(num_classes=10, img_size=(32,32)):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, img_size=img_size)


def resnet152(num_classes=10, img_size=(32,32)):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, img_size=img_size)
