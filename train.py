import sys,time,os,torch,argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from model import UNet,TransUNet,SwinUnet,deeplabv3p,m_segformer
from utils import train_Dataset,log_output,fast_hist,per_class_iu,per_class_PA_Recall,per_class_Precision
from utils import adamw,ExponentialLR,CosLR,CrossEntropy_Loss,focal_loss

def get_arguments():

    parser = argparse.ArgumentParser(description="pytorch Network")
    ###### ----------- 数据集相关设置 -------------- ######
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, help="datasets path",
                        default="D:\\software\\Code\\codefile\\result\\mydata\\model_test_data")#D:\\software\\Code\codefile\\image_result\\mydata\\model_test_data
    parser.add_argument("--save_dir", type=str,help="save path .",
                        default="D:\\software\\Code\\codefile\\mseg\\results")#D:\\software\\Code\\codefile\\mseg\\results
    parser.add_argument("--input_size", type=list, default=[512,512],
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--traindata_rate", type=int, default=0.9,
                        help="Proportion of training datasets.") 
    
    ###### ------------ 设置模型 --------------- ######
    parser.add_argument("--arch", type=str, default="m_segformer", 
                        help="[UNet, pspnet_smp, TransUNet, SwinUnet, DualSeg_res101, deeplabv3p, m_segformer]")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--name_classes", type=list, default=["background", "root"],
                        help="Name of classes to predict (including background).")
    
    ###### ------------ 模型训练相关设置 ----------- ######
    parser.add_argument("--epochs", type=int, default=100,
                        help="train epochs.")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="[SGD, adam ,adamw,RMSprop].")
    parser.add_argument("--lr_scheduler", type=str, default="CosLR",
                        help="[ExponentialLR,CosLR].")
    parser.add_argument("--criterion", type=str, default="focal_loss",
                        help="[CriterionOhemDSN, CriterionDSN ,CrossEntropy_Loss,focal_loss].")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Regularisation parameter for L2-loss.")
    
    ###### ------------ 其他设置 ----------------- ######
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")
    args = parser.parse_args()
    return args

def main(config):

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #重新设置保存路径，在原始路径下添加时间文件夹
    save_pth = os.path.join(config.save_dir,config.arch,str(time.localtime()[1])+"-"+str(time.localtime()[2])
                            +"-"+str(time.localtime()[3])+"-"+str(time.localtime()[4]))
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    
    # log文件设置
    logger = log_output(save_pth)

    # 加载训练集(划分训练和验证机的比例)
    all_dataset = train_Dataset(config.data_dir,config.input_size)
    train_size = int(config.traindata_rate * len(all_dataset))  ## 训练集大小
    test_size = len(all_dataset) - train_size                   ## 验证集大小
    train_dataset, trainval_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size]) 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)
    
    #加载模型
    net = eval(config.arch)(config.num_classes).to(device=device)
    # 定义优化器算法
    optimizer = eval(config.optimizer)(net, lr=config.lr)
    # 学习率优化算法
    scheduler = eval(config.lr_scheduler)(optimizer)
    # 定义Loss算法
    criterion = eval(config.criterion)

    # 打印各种参数
    logger.info('Segformer_primary + 改善最后的解码头 +cos +crossloss +adaw')
    logger.info('epochs: {}, batch_size: {}, lr: {}'.format(config.epochs,config.batch_size,config.lr))
    logger.info(net)
    logger.info('optimizer:{}'.format(config.optimizer))
    logger.info(optimizer.state_dict())
    logger.info(config.lr_scheduler)
    logger.info(criterion)

    #初始化混合精度
    scaler = GradScaler()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(config.epochs):
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
                        .format(epoch+1,batch_idx*config.batch_size,len(train_dataset),100.0*(batch_idx*config.batch_size)/len(train_dataset),loss.item()))
            # 保存loss值最小的网络参数
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(net.state_dict(), os.path.join(save_pth,'last_model.pth'))
            # 更新参数
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        #学习率参数更新
        scheduler.step()

        #模型验证
        if len(trainval_dataset) !=0 and (epoch+1) % 2==0:
            net.eval()
            with torch.no_grad():
                hist = np.zeros((config.num_classes, config.num_classes))
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

                    hist += fast_hist(label, pred, config.num_classes)
                IoUs = per_class_iu(hist)
                PA_Recall = per_class_PA_Recall(hist)
                Precision = per_class_Precision(hist)
                for ind_class in range(config.num_classes):
                    logger.info('---Trian val---' + config.name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                    + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                    round(Precision[ind_class] * 100, 2)))
                    
    logger.info('---Trian finish--- ')
            
if __name__ == "__main__":

    config = get_arguments()
    main(config)