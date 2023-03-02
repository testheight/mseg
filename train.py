import sys,time,os,torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from model.unet_model import UNet
from model.transunet import TransUNet
from model.swin_unet import SwinUnet
from utils import train_Dataset,log_output,fast_hist,per_class_iu,per_class_PA_Recall,per_class_Precision



def train_net(net, device, data_path, save_pth, epochs=40, batch_size=2, lr=0.001, num_classes=2):

    #实际类别
    name_classes = ["background", "root"]
    #重新设置保存路径，在原始路径下添加时间文件夹
    save_pth = os.path.join(save_pth,str(time.localtime()[1])+"-"+str(time.localtime()[2])
                            +"-"+str(time.localtime()[3])+"-"+str(time.localtime()[4]))
    # 检测保存路径是否存在
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    # log文件设置
    logger = log_output(save_pth)
    logger.info('epochs: {}, batch_size: {}, lr: {}'.format(epochs,batch_size,lr))
    logger.info(net)
    # 加载训练集
    all_dataset = train_Dataset(data_path)
    # 设置划分比例
    train_size = int(0.9 * len(all_dataset))  # 整个训练集中，百分之90为训练集
    test_size = len(all_dataset) - train_size
    # 划分训练集和测试集
    train_dataset, trainval_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size]) 
    # 加载数据集 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 学习率优化算法
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.5)
    # 定义Loss算法
    criterion = torch.nn.CrossEntropyLoss()
    #初始化混合精度
    scaler = GradScaler()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
            # 训练模式
        net.train()
            # 按照batch_size开始训练
        for batch_idx,(image, label) in enumerate(train_loader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            #混合精度训练
            with autocast():
                # 使用网络参数，输出预测结果#(pred.shape=1,2,640,640, label=1,640,640)
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
            #输出log
            logger.info('---Trian--- epoch: {} [{}/{} ({:.0f}%)], loss: {:.6f} '
                        .format(epoch+1,batch_idx*batch_size,len(train_dataset),100.0*(batch_idx*batch_size)/len(train_dataset),loss.item()))
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), os.path.join(save_pth,'last_model.pth'))
            # 更新参数
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        #学习率参数更新
        scheduler.step()

        #模型验证
        if len(trainval_dataset) !=0 and (epoch+1) % 5==0:
            net.eval()
            with torch.no_grad():
                hist = np.zeros((num_classes, num_classes))
                for i in tqdm(range(len(trainval_dataset))):
                    img, label = trainval_dataset[i]
                    img = img.unsqueeze(0)                                                  # --------设置图像尺寸-----#
                    img = img.to(device=device, dtype=torch.float32)                        # ---tensor拷贝到device中--#
                    pred = net(img)
                    #(pred.shape=1,2,640,640, label=1,640,640)
                    pred = (torch.nn.functional.softmax(pred[0],dim=0)).data.cpu()
                    pred = np.array(pred.argmax(axis=0))
                    label = np.array(label)


                    if len(label.flatten()) != len(pred.flatten()):
                        print('Skipping: 预测和标签数据不相等')
                        continue
                    
                    label = np.array([int(x) for x in label.flatten()]) 
                    pred = np.array([int(x) for x in pred.flatten()])  

                    hist += fast_hist(label, pred, num_classes)
                IoUs = per_class_iu(hist)
                PA_Recall = per_class_PA_Recall(hist)
                Precision = per_class_Precision(hist)
                for ind_class in range(num_classes):
                    logger.info('---Trian val---' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                    + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                    round(Precision[ind_class] * 100, 2)))
                    
    logger.info('---Trian finish--- ')
            



if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络。
    # net = UNet(n_channels=3, n_classes=2)
    # net = TransUNet(img_dim=512,
    #                       in_channels=3,
    #                       out_channels=128,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       class_num=2)
    net = SwinUnet(
                img_size=512, 
                num_classes=2
    )

    # 将网络拷贝到deivce中
    # net.to(device=device)
    # # 指定训练集地址，开始训练
    # data_path = r"/home/lijiangtao/Desktop/T/mseg/datasets/root_data"       # todo 修改为你本地的数据集位置
    # save_pth = r'/home/lijiangtao/Desktop/T/mseg/results/transunet'    # 权重保存路径
    # train_net(net, device, data_path, save_pth, epochs=100, batch_size=8, lr=0.000001)
