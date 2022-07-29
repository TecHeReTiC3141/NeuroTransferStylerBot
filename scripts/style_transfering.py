from __future__ import print_function

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

import requests
from io import BytesIO

import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS = torch.load('model_weights/vgg19.pth')

IMG_SIZE = 256


def get_image(img_url: str) -> BytesIO:
    return BytesIO(requests.get(img_url).content)


def image_loader(image_url, imsize=IMG_SIZE, no_resize=False, norm=False) -> torch.tensor:
    loader = transforms.Compose([
        transforms.Resize(imsize) if not no_resize else nn.Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if norm else nn.Identity()
    ])
    image = Image.open(get_image(image_url))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor: torch.tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def content_loss(input, target):
    return F.mse_loss(input, target)


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    return G.div(a * b * c * d)


def style_loss(input, target):
    input = gram_matrix(input)
    target = gram_matrix(target)
    return F.mse_loss(target, input)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class vgg(nn.Module):
    features = [0, 5, 10, 19, 28]

    def __init__(self, cnn: nn.Module, cnn_mean, cnn_std):
        super().__init__()
        # self.model = nn.Sequential(Normalization(cnn_mean, cnn_std))
        # for layer in range(len(cnn[:self.features[-1] + 1])):
        #     self.model.add_module(str(layer), cnn[layer])

        norm = Normalization(cnn_mean, cnn_std)
        seq = [norm]
        seq.extend(list(cnn[:self.features[-1] + 1].children()))
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if layer_num in self.features:  # Если слой находится в списке,
                features.append(x)  # то запоминаем выход слоя для расчета лосса

        return features


def img_to_bytes(img: Image, imsize=IMG_SIZE) -> BytesIO:
    byte_arr = BytesIO()
    width, height = img.size
    if width > height:
        img_res = img.resize((imsize, int(imsize * height / width)))
    else:
        img_res = img.resize((int(imsize * width / height), imsize))
    out_dir = Path('outputs')

    img_res.save(out_dir / 'result.png', format='PNG')
    img_res.save(byte_arr, format='PNG')

    return byte_arr.getvalue()


class StyleTransfer(nn.Module):

    def __init__(self, cnn: nn.Sequential,
                 normalization_mean: torch.tensor, normalization_std: torch.tensor, ):
        super().__init__()
        self.cnn = cnn
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def forward(self, content_img_url: str, style_img_url: str, weights=WEIGHTS, num_steps=300,
                style_weight=1000000, content_weight=1) -> BytesIO:

        try:
            content_img = image_loader(content_img_url)
            style_img = image_loader(style_img_url)
            input_img: torch.tensor = content_img.clone()

        except Exception as e:
            print(e)
            return "Error"

        model = vgg(self.cnn,
                    self.normalization_mean,
                    self.normalization_std)

        model.load_state_dict(weights)

        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = optim.LBFGS([input_img])

        content_features = model(content_img)
        style_features = model(style_img)

        run = [0]
        try:
            with torch.autograd.set_detect_anomaly(True):
                while run[0] <= num_steps:
                    def closure():
                        with torch.no_grad():
                            input_img.clamp_(0, 1)

                        optimizer.zero_grad()

                        style_score = 0
                        content_score = 0

                        generated_features = model(input_img.clone())

                        for gen_features, con_features, st_features in zip(
                                generated_features,
                                content_features,
                                style_features
                        ):
                            content_score += content_loss(con_features, gen_features)
                            style_score += style_loss(st_features, gen_features)

                        style_score *= style_weight
                        content_score *= content_weight
                        loss = style_score + content_score
                        loss.backward()

                        run[0] += 1
                        if run[0] % 100 == 0:
                            print(
                                f'epoch: {run[0]}, style_loss: {style_score.item():.4f}, content_loss: {content_score.item():.4f}')
                        return style_score + content_score

                    optimizer.step(closure)

            # imshow(input_img)
            with torch.no_grad():
                input_img.clamp_(0, 1)

            res = unloader(input_img.cpu().squeeze(0))
            return img_to_bytes(res)
        except Exception as e:
            return e


cnn = models.vgg19().features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
