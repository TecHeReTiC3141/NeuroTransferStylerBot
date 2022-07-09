from slow_algorithm_of_style_transfering import imshow, \
    img_to_bytes, unloader, plt

import torch

tensor1, tensor2 = torch.load('../outputs/tensor0.pt'), \
                   torch.load('../outputs/tensor1.pt')
#
# imshow(tensor1)
# imshow(tensor2)
img1 = unloader(tensor1.clone().detach().squeeze(0))
img2 = unloader(tensor2.clone().detach().squeeze(0))

plt.imshow(img1)
plt.imshow(img2)