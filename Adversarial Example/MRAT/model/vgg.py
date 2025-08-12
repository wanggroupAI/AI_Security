from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, img_size = (32,32), init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # whether to use mat or not
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

        self.features = features
        self.decoder = nn.Sequential(*decoder_lis)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.features(x)
        x2 = self.avgpool(x1)
        x3 = torch.flatten(x2, 1)
        x4 = self.classifier(x3)
        if self.is_mask:
            reimage = self.decoder(x1)
            return x4, reimage
        else:
            return x4

    def turn_on_mask(self):
        self.is_mask = True
        
    def turn_off_mask(self):
        self.is_mask = False
        
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights, progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model


def vgg11(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("A", False, weights, progress, **kwargs)



def vgg11_bn(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:

    return _vgg("A", True, weights, progress, **kwargs)



def vgg13(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("B", False, weights, progress, **kwargs)



def vgg13_bn(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("B", True, weights, progress, **kwargs)



def vgg16(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("D", False, weights, progress, **kwargs)



def vgg16_bn(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("D", True, weights, progress, **kwargs)


def vgg19(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("E", False, weights, progress, **kwargs)
    
def vgg19_bn(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("E", True, weights, progress, **kwargs)