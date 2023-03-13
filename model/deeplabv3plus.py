import torch
import segmentation_models_pytorch as smp

def deeplabv3p(num_classes=2):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,                      # model output channels (number of classes in your dataset)
    )
    return model

if __name__ == "__main__":
    a = torch.rand(2,3,512,512)
    net = deeplabv3p()
    mask = net(a)
    print(mask.shape)