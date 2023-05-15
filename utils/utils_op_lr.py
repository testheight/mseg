import torch,os,tqdm
from PIL import Image

#优化器
def SGD(net,lr=0.01):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9,weight_decay=0.0005,eps=1e-5)
    return optimizer

def adam(net,lr=0.01,betas=(0.9,0.999)):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas,eps=1e-5)
    return optimizer

def adamw(net,lr=0.01,betas=(0.9,0.999)):
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=betas,eps=1e-5)
    return optimizer

def RMSprop(net,lr=0.01,weight_decay=1e-8, momentum=0.9):
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum,eps=1e-5)
    return optimizer

#学习率优化算法
def ExponentialLR(optimizer,gamma=0.8):
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
    return scheduler

# 余弦退火
def CosLR(optimizer):
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,)
    return scheduler

# 创建学习率更新策略，这里是每个step更新一次学习率，以及使用warmup
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def Stitching_images(image_file,Stitching_file,pixel):
    '''     img_file:拼接图片的路径
            Stitching_file:保存的路径  '''
    if not os.path.exists(Stitching_file):
        os.makedirs(Stitching_file)

    pixel_wide = pixel_high =pixel
    name_list = []
    for name in os.listdir(image_file):
        if name.split('_')[0] not in name_list:
            name_list.append(name.split('_')[0])

    for name in tqdm.tqdm(name_list):
        row_list = []
        column_list = []
        for file in os.listdir(image_file):
            if name in file:
                if file.split("_")[1] not in row_list:
                    row_list.append((file.split("_")[1]))
                if file.split("_")[2].split('.')[0] not in column_list:
                    column_list.append((file.split("_")[2].split('.')[0]))

        row = len(row_list)
        column = len(column_list)
        #创建空白画布
        target = Image.new('RGB', (pixel_wide * column, pixel_high * row))  
        #size: A 2-tuple, containing (width, height) in pixels.
        for i in range(row):
            for j in range(column):
                image = Image.open(
                    os.path.join(image_file,name+"_"+str(i+1)
                                     +"_"+str(j+1)+".png"))
                target.paste(image, (j * pixel_wide, i * pixel_high))
        target.save(Stitching_file+"\\"+str(name)+".png")