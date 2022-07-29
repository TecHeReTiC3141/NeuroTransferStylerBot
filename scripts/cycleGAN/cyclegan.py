import torchvision.utils

from scripts.style_transfering import *
from scripts.cycleGAN.cycleGAN_generator import Generator

H_to_Z_weights = torch.load('scripts/cycleGAN/weights/netG_A2B.pth', map_location=device)
Z_to_H_weights = torch.load('scripts/cycleGAN/weights/netG_B2A.pth', map_location=device)


class CycleGAN(nn.Module):

    def __init__(self, trans_type: str):
        super().__init__()

        self.gen = Generator().to(device).eval()
        assert trans_type in ['h2z', 'z2h'], 'Wrong transform mode'
        if trans_type == 'h2z':
            self.gen.load_state_dict(H_to_Z_weights)
        else:
            self.gen.load_state_dict(Z_to_H_weights)

    def forward(self, img_url):
        img = image_loader(img_url, imsize=512, norm=True)
        gen_img = self.gen(img)
        grid = torchvision.utils.make_grid(gen_img.detach(), normalize=True)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

        return img_to_bytes(Image.fromarray(ndarr))
