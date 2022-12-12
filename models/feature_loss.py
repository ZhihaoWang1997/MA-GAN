
import torch
import torch.nn as nn
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self, state_dict):
        super(FeatureExtractor, self).__init__()
        self.extractor = torchvision.models.resnet18(pretrained=False)
        self.extractor.load_state_dict(torch.load(state_dict))

    def forward(self, x):
        x = self.extractor.conv1(x)
        x = self.extractor.bn1(x)
        x = self.extractor.relu(x)
        x = self.extractor.maxpool(x)

        x = self.extractor.layer1(x)
        x = self.extractor.layer2(x)
        x = self.extractor.layer3(x)
        x = self.extractor.layer4(x)

        x = self.extractor.avgpool(x)
        return x


if __name__ == '__main__':
    a = FeatureExtractor()
    print(a)
