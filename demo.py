import os,torch,tqdm,cv2
import numpy as np
from model import unet_smp
from utils import demo_dataset

loda_path = r"D:\31890\Desktop\codefile\result\mseg_result\2\unet_smp\3-24-17-57\last_model.pth"
test_dir = r"D:\31890\Desktop\codefile\mseg\image"
save_patjh = r"D:\31890\Desktop\codefile\mseg\image2"

if not os.path.exists(save_patjh):
    os.makedirs(save_patjh)

testdata = demo_dataset(test_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # --------选择容器-------#
net = unet_smp(num_classes = 2)                                         # --------加载网络-------#
net.to(device=device)                                                       # ---将网络拷贝到deivce中--#
state_dict = torch.load(loda_path, map_location=device)
net.load_state_dict(state_dict) # 从新加载这个模型。


net.eval()
with torch.no_grad():
    for i in tqdm.tqdm(range(len(testdata))):
        img, id = testdata[i]

        img = img.unsqueeze(0)                                                  # --------设置图像尺寸-----#
        img = img.to(device=device, dtype=torch.float32)                        # ---tensor拷贝到device中--#

        pred = net(img)                                                         # --------预测图像---------#

        pred = (torch.nn.functional.softmax(pred[0],dim=0)).data.cpu()          # --------转换为array------#
        pred = np.array(pred.argmax(axis=0))*255

        cv2.imwrite(os.path.join(save_patjh,id+".png"),pred) 
    