import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet201, squeezenet1_0


def prepare_model(model_params):
    """Initializes the model from the dictionary of parameters.
    Arguments:
        model_params: dict with model configuration
    Returns:
        model: nn.Module that contains the model
    """
    model_arch = model_params.get('model', None)
    if model_arch == 'SqueezeNet':
        model = SqueezeNetModel()
    else:
        model = DenseNet201Model()
    
    return model


class DenseNet201Model(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = densenet201(pretrained=True).features
        last_module = list(self.features.modules())[-1]
        self.classifier = nn.Linear(last_module.num_features, 7)
    
    def forward(self, x):
        images = x['image']
        features = self.features(images)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        
        return out


class SqueezeNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = squeezenet1_0(pretrained=True).features
        
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )
        nn.init.normal(final_conv.weight.data, mean=0.0, std=0.01)
    
    def forward(self, x):
        images = x['image']
        x = self.features(images)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
