from scripts.slow_algorithm_of_style_transfering import *

origin_url, style_url = input(), input()

origin_image, style_image = image_loader(origin_url), image_loader(style_url)

imshow(origin_image)

imshow(style_image)