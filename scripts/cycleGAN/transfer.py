import torchvision.utils

from scripts.style_transfering import *
from cycleGAN_generator import Generator

H_to_Z_weights = torch.load('scripts/cycleGAN/weights/netG_A2B.pth')
Z_to_H_weights = torch.load('scripts/cycleGAN/weights/netG_B2A.pth')

class CycleGAN(nn.Module):

    def __init__(self, trans_type: str):
        super().__init__()

        self.gen = Generator().to(device).eval()
        assert trans_type in ['h2z', 'z2h'], 'Wrong transform mode'
        if trans_type == 'h2z':
            self.gen.load(H_to_Z_weights)
        else:
            self.gen.load(Z_to_H_weights)

    def forward(self, img_url):
        img = image_loader(img_url, norm=True)
        gen_img = self.gen(img)
        grid = torchvision.utils.make_grid(gen_img)
        grid =  grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0) .to("cpu", torch.uint8).numpy()
        imshow(grid)
        res = Image.fromarray(grid)
        return img_to_bytes(res)
