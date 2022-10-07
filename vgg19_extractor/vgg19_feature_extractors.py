import torch
from torchvision import transforms


def vgg19_conv1_2_feature_extractor():
    """
    produce a torch.nn.Sequential module that consists up to the conv1_2 (inclusive) of the VGG19 network, with frozen parameters

    :return: torch.nn.Sequential of VGG19 conv1_2 feature extractor
    """
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    vgg19 = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True)
    ret = torch.nn.Sequential(*list(list(vgg19.children())[0].children())[0:3])
    for params in ret.parameters():
        params.requires_grad = False

    return ret


def vgg19_image_preproc(pil_images):
    """
    preprocesses images in order to match the input requirement of VGG19 network

    :param pil_images: a single PIL Image or a list of PIL Image's
    :return: pytorch Tensor of shape (N,C,H,W), where N is 1 when pil_images is a single image and N is the length of the list otherwise
    """
    preprocessor = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # this also crops image into the range [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if type(pil_images) is list:
        tensor_list = [torch.unsqueeze(preprocessor(x), 0) for x in pil_images]
        ret = torch.cat(tensor_list, 0)
    else:
        ret = torch.unsqueeze(preprocessor(pil_images), 0)

    return ret
