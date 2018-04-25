import torch.nn
import torch.nn.functional as F

from torchvision.models import densenet201, squeezenet1_0


def prepare_model(model_params):
    model = DenseNet201Model()
    
    return model


class DenseNet201Model(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = densenet201(pretrained=True).features
        last_module = list(self.features.modules())[-1]
        self.classifier = torch.nn.Linear(last_module.num_features, 7)
    
    def forward(self, x):
        images = x['image']
        features = self.features(images)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        
        return out


class SqueezeNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_0(pretrained=True)
    
    def forward(self, x):
        images = x['image']
        result = self.model.forward(images)
        
        return result
