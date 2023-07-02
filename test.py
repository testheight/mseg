import os,cv2,torch
from tqdm import tqdm
from utils import compute_mIoU, show_results,test_Dataset,demo_dataset
import numpy as np
from model import U_Net_o,U_Net_dws,U_Net_dws2,U_Net_top2bottom
from utils import Stitching_images
from PIL import Image


def cal_miou(test_dir,result_dir):                      # ---图像测试集路径和标签路径----#

    num_classes = 2                                     # -----------分类个数----------#
    name_classes = ["background", "root"]               # -----------实际类别----------#
    testdata = test_Dataset(test_dir)                   # -----------加载数据集--------#

    if not os.path.exists(os.path.join(result_dir,os.path.basename(test_dir),'mask')):     # -----检测图像保存文件夹是否存在----#
        os.makedirs(os.path.join(result_dir,os.path.basename(test_dir),'mask'))
    if not os.path.exists(os.path.join(result_dir,os.path.basename(test_dir),'metric')):
        os.makedirs(os.path.join(result_dir,os.path.basename(test_dir),'metric'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # --------选择容器-------#
    net = U_Net_top2bottom(num_classes = num_classes)                                         # --------加载网络-------#
    net.to(device=device)                                                       # ---将网络拷贝到deivce中--#
    net.load_state_dict(torch.load(os.path.join(result_dir,"min_loss_model.pth"),   # ------加载模型参数-------#
                                       map_location=device)) 
    # net.load_state_dict(torch.load(os.path.join(result_dir,"epoch_200_model.pth"),   # ------加载模型参数-------#
    #                                    map_location=device)) 
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


            cv2.imwrite(os.path.join(result_dir,os.path.basename(test_dir),'mask',id+".png"),pred)             # --------保存图像---------#

    print("Get predict result done.")

    gt_dir = os.path.join(test_dir,'anno','test')
    pred_dir = os.path.join(result_dir,os.path.basename(test_dir),'mask')
    if os.path.basename(test_dir)=='mseg_root_data':
        Stitching_dir = os.path.join(result_dir,os.path.basename(test_dir),'mask_large')
        Stitching_images(pred_dir,Stitching_dir,512)

    print("Stitch images done.")
    
    hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes,name_classes)              # -----执行计算mIoU的函数---#
    
    print("Get miou done.")

    if not os.path.exists(os.path.join(result_dir,os.path.basename(test_dir),'metric')):
        os.makedirs(os.path.join(result_dir,os.path.basename(test_dir),'metric'))
    show_results(os.path.join(result_dir,os.path.basename(test_dir),'metric'), hist, IoUs,                 # -----生成mIoU的图像------#
                 PA_Recall, Precision, name_classes)


def infer(para_path,test_dir,save_path):
    num_classes = 2
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    testdata = demo_dataset(test_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # --------选择容器-------#
    net = U_Net_o(num_classes = num_classes)     #unet_smp  segformer_m                                    # --------加载网络-------#
    net.to(device=device)                                                       # ---将网络拷贝到deivce中--#
    state_dict = torch.load(para_path, map_location=device)
    net.load_state_dict(state_dict) # 从新加载这个模型。

    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(testdata))):
            img, id = testdata[i]

            img = img.unsqueeze(0)                                                  # --------设置图像尺寸-----#
            img = img.to(device=device, dtype=torch.float32)                        # ---tensor拷贝到device中--#

            pred = net(img)                                                         # --------预测图像---------#

            pred = (torch.nn.functional.softmax(pred[0],dim=0)).data.cpu()          # --------转换为array------#
            if num_classes==2:
                pred = np.array(pred.argmax(axis=0))*255
                cv2.imwrite(os.path.join(save_path,id+".png"),pred)
            else:
                pred = np.array(pred.argmax(axis=0)).astype(np.int8)

                pred = Image.fromarray(pred)
                pred = pred.convert('L')
                #调色板
                # palette = [0, 0, 0,0, 255, 0, 255, 0, 0,255,255,255]
                palette = [0, 0, 0,255,255,255]
                #着色
                pred.putpalette(palette)
                pred.save(os.path.join(save_path,id+".png"))

# 预测大图
def infer2(para_path,test_dir,save_path,num_classes = 2,pixel_shape=512):
    num_classes = 2
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    testdata = demo_dataset(test_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # --------选择容器-------#
    net = U_Net_o(num_classes = num_classes)                                   # --------加载网络-------#
    net.to(device=device)                                                       # ---将网络拷贝到deivce中--#
    state_dict = torch.load(para_path, map_location=device)
    net.load_state_dict(state_dict) # 从新加载这个模型。

    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(testdata))):
            img, id = testdata[i]
            img = img.permute(1,2,0)
            img_array = img.numpy()
            h,w,c = img_array.shape
            if h//pixel_shape!=0 and w//pixel_shape!=0:
                h_n = (h//pixel_shape+1)*pixel_shape
                w_n = (w//pixel_shape+1)*pixel_shape
                h_padding = h_n-h
                w_padding = w_n-w
            if h//pixel_shape==0:
                h_n = pixel_shape
                h_padding = h_n-h
            if w//pixel_shape==0:
                w_n = pixel_shape
                w_padding = w_n-w
            if h//pixel_shape!=0:
                h_n = (h//pixel_shape+1)*pixel_shape
                h_padding = h_n-h
            if w//pixel_shape!=0:
                w_n = (w//pixel_shape+1)*pixel_shape
                w_padding = w_n-w

            img2 = cv2.copyMakeBorder(img_array,0,h_padding,0,w_padding,cv2.BORDER_CONSTANT)
            img3 = torch.from_numpy(img2)
            img3 = img3.permute(2,0,1)
            img3 = img3.unsqueeze(0) 
            pred_result = torch.from_numpy(np.empty((1,num_classes,h_n,w_n)))

            for u in range(h_n//pixel_shape):
                for v in range(w_n//pixel_shape):
                    x = pixel_shape * u
                    y = pixel_shape * v
                    sub_img = img3[:,:,x : x + pixel_shape, y : y + pixel_shape]
                    sub_img = sub_img.to(device=device, dtype=torch.float32)                        # ---tensor拷贝到device中--#
                    pred = net(sub_img)                                                         # --------预测图像---------#
                    pred_result[:,:,x : x + pixel_shape, y : y + pixel_shape] = pred.cpu()

            pred_result = (torch.nn.functional.softmax(pred_result[0],dim=0)).data.cpu()          # --------转换为array------#
            pred_result = pred_result[:,0:h,0:w]
            
            if num_classes==2:
                pred_result = np.array(pred_result.argmax(axis=0))*255
                cv2.imwrite(os.path.join(save_path,id+".png"),pred_result)
            else:
                pred_result = np.array(pred_result.argmax(axis=0)).astype(np.int8)

                pred_result = Image.fromarray(pred_result)
                pred_result = pred_result.convert('L')
                #调色板
                palette = [0, 0, 0,0, 255, 0, 255, 0, 0,255,255,255]
                #着色
                pred.putpalette(palette)
                pred.save(os.path.join(save_path,id+".png"))

if __name__ == '__main__':

    # cal_miou(test_dir = r"D:\software\Code\codefile\result\mydata\model_test_data",         #测试数据集
    #          result_dir=r"D:\software\Code\codefile\mseg\results\unet_smp\3-25-18-34")
    
    # infer(para_path =  r"D:\31890\Desktop\codefile\data\mseg_result\2\unet_smp\3-24-17-57\last_model.pth",
    #       test_dir=r"D:\31890\Desktop\codefile\data\Train_data\Datasets\1-100_o",
    #       save_path=r"D:\31890\Desktop\codefile\mseg\result\test")
    
    infer2(para_path =  r"D:\31890\Desktop\codefile\data\mseg_result\2\unet_smp\3-24-17-57\last_model.pth",
          test_dir=r"D:\31890\Desktop\codefile\data\Train_data\Datasets\1-100_o",
          save_path=r"D:\31890\Desktop\codefile\mseg\result\test")