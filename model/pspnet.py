import torch
import segmentation_models_pytorch as smp

def pspnet_smp(num_classes=2):
    model = smp.PSPNet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,                      # model output channels (number of classes in your dataset)
    )
    return model

if __name__ == "__main__":
        
    from thop import profile
    net = pspnet_smp()
    x = torch.randn(2, 3, 512, 512)
    flops , params = profile(net,inputs=(x,))
    print(flops)
    print(params)
