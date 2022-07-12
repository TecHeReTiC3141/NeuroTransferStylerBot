from scripts.slow_algorithm_of_style_transfering import *

transfer = StyleTransfer(cnn, cnn_normalization_mean, cnn_normalization_std)

output = transfer('../images/2BDamned.jpg', '../images/colorful_image.jpg')
