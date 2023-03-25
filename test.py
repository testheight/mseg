import os,cv2,torch
from tqdm import tqdm
from utils import compute_mIoU, show_results,test_Dataset
import numpy as np
from model import transunet_m,swinunet_m,deeplabv3p_smp,unet_smp,pspnet_smp,segnet_m
from utils import Stitching_images


def cal_miou(test_dir,result_dir):                      # ---图像测试集路径和标签路径----#

    num_classes = 2                                     # -----------分类个数----------#
    name_classes = ["background", "root"]               # -----------实际类别----------#
    testdata = test_Dataset(test_dir)                   # -----------加载数据集--------#

    if not os.path.exists(os.path.join(result_dir,'mask')):     # -----检测图像保存文件夹是否存在----#
        os.makedirs(os.path.join(result_dir,'mask'))
    if not os.path.exists(os.path.join(result_dir,'metric')):
        os.makedirs(os.path.join(result_dir,'metric'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # --------选择容器-------#
    net = unet_smp(num_classes = num_classes)                                         # --------加载网络-------#
    net.to(device=device)                                                       # ---将网络拷贝到deivce中--#
    net.load_state_dict(torch.load(os.path.join(result_dir,"last_model.pth"),   # ------加载模型参数-------#
                                       map_location=device)) 
    
    image_ids=[]
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(testdata))):
            img, label, id = testdata[i]
            image_ids.append(id)

            img = img.unsqueeze(0)                                                  # --------设置图像尺寸-----#
            img = img.to(device=device, dtype=torch.float32)                        # ---tensor拷贝到device中--#

            pred = net(img)                                                         # --------预测图像---------#

            pred = (torch.nn.functional.softmax(pred[0],dim=0)).data.cpu()          # --------转换为array------#
            pred = np.array(pred.argmax(axis=0))*255


            cv2.imwrite(os.path.join(result_dir,'mask',id+".png"),pred)             # --------保存图像---------#

    print("Get predict result done.")

    gt_dir = os.path.join(test_dir,'anno','test')
    pred_dir = os.path.join(result_dir,'mask')
    Stitching_dir = os.path.join(result_dir,'mask_large')

    Stitching_images(pred_dir,Stitching_dir,512)    
    print("Stitch images done.")
    
    hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes,name_classes)              # -----执行计算mIoU的函数---#
    
    print("Get miou done.")

    if not os.path.exists(os.path.join(result_dir,'metric')):
        os.makedirs(os.path.join(result_dir,'metric'))
    show_results(os.path.join(result_dir,'metric'), hist, IoUs,                 # -----生成mIoU的图像------#
                 PA_Recall, Precision, name_classes)

if __name__ == '__main__':

    cal_miou(test_dir = r"D:\software\Code\codefile\result\mydata\model_test_data",         #测试数据集
             result_dir=r"D:\software\Code\codefile\mseg\results\unet_smp\3-25-18-34")