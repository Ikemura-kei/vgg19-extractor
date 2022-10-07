import torch
from PIL import Image
import cv2
import os
import numpy as np
from os import listdir
from vgg19_extractor.vgg19_feature_extractors import vgg19_conv1_2_feature_extractor
from vgg19_extractor.vgg19_feature_extractors import vgg19_image_preproc


def main():
    print("pytorch version:", torch.__version__)
    print("cuda version:", torch.version.cuda)

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("using GPU, GPU device name:", torch.cuda.get_device_name(0))
    else:
        print("using CPU")
        dev = "cpu"

    torch_device = torch.device(dev)

    image_path = "./sample_images"
    images = []
    for f in listdir(image_path):
        if f.endswith("jpg"):
            file_name = os.path.join(image_path, f)
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            images.append(im)

    images = [Image.fromarray(im) for im in images]

    image_input = vgg19_image_preproc(images)
    print("prepared input shape", image_input[0].shape)

    extractor = vgg19_conv1_2_feature_extractor()

    image_input = image_input.to(torch_device)
    extractor = extractor.to(torch_device)

    feature = extractor(image_input)

    print("feature size:", feature.shape)

    print("checking if parameters are really freezed:")
    for p in extractor.parameters():
        print("\tfreezed:", not(p.requires_grad))
    
    print("\nextractor architecture:")
    print(extractor)


if __name__ == "__main__":
    main()
