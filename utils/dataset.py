import torch,os,cv2
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import random_split

class train_Dataset(Dataset):
    def __init__(self,data_dir):
        #建立数据列表
        images,labels = [],[]
        for name in os.listdir(os.path.join(data_dir,'imgs','train')):
                images.append(os.path.join(data_dir,'imgs','train',name))
                labels.append(os.path.join(data_dir,'anno','train',name.split('.')[0]+'.png'))
        self.labels = labels 
        self.images = images

    #获取图像    
    def __getitem__(self, index):
        img_path,label_path =  self.images[index],self.labels[index]
        #读取图像
        imgs = cv2.imread(img_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        lbls = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        #数据增强
        transform = A.Compose([
                A.RandomResizedCrop(height=512,width=512,scale=(0.15, 1.0)),    #旋转
                A.Rotate(p=0.3),                                                #翻转
                A.HorizontalFlip(p=0.3),                                        #水平翻转
                A.VerticalFlip(p=0.2),                                          #垂直翻转
                A.OneOf([
                    #随机改变图像的亮度、对比度和饱和度
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.1),
                    #随机改变输入图像的色调、饱和度和值
                    A.HueSaturationValue(p=0.3),
                    ],p=0.2),
                A.Normalize (mean=[0.4754358, 0.35509014, 0.282971],std=[0.16318515, 0.15616792, 0.15164918]),
                ToTensorV2(),
                ])
        transformed = transform(image=imgs, mask=lbls)
        imgs = transformed['image']
        lbls = transformed['mask'].long()
        return imgs,lbls
    
    #获取数据集长度
    def __len__(self):
        return len(self.images)

class test_Dataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        #建立数据列表
        images,labels = [],[]
        for name in os.listdir(os.path.join(data_dir,'imgs','test')):
                images.append(os.path.join(data_dir,'imgs','test',name))
                labels.append(os.path.join(data_dir,'anno','test',name.split('.')[0]+'.png'))
        self.labels = labels 
        self.images = images
        
    def __getitem__(self, index):
        #读取图像
        img_path,label_path =  self.images[index],self.labels[index]
        imgs_id = img_path.split("/")[-1].split(".")[0]
        imgs = cv2.imread(img_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        lbls = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        #数据增强
        transform = A.Compose([
                A.Normalize (mean=[0.4754358, 0.35509014, 0.282971],std=[0.16318515, 0.15616792, 0.15164918]),
                ToTensorV2(),
            ])
        transformed = transform(image=imgs, mask=lbls)
        imgs = transformed['image']
        lbls = transformed['mask'].long()
        return imgs,lbls,imgs_id
    
    def __len__(self):
        return len(self.images)

def test_data():
    data_dir = r'/home/lijiangtao/Desktop/T/mseg/datasets/root_data'
    datas = train_Dataset(data_dir)



    train_size = int(0.9 * len(datas))  # 整个训练集中，百分之90为训练集
    test_size = len(datas) - train_size
    train_dataset, test_dataset = random_split(datas, [train_size, test_size])  # 划分训练集和测试集

    print(len(train_dataset))
    print(len(test_dataset))

    # mydataloader = DataLoader(train_dataset, batch_size=4, shuffle=False,num_workers=1)
    # img_tensor1,img_tensor2 = next(iter(mydataloader))
    # # show(batch_size,img_tensor1,img_tensor2)
    # print(img_tensor1.shape)
    # print(img_tensor2.shape)

if __name__ == "__main__":
    # isbi_dataset = Root_Dataset(r"D:\software\Code\codefile\image_result\mydata\my_data")
    # print("数据个数：", len(isbi_dataset))
    # train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
    #                                            batch_size=2, 
    #                                            shuffle=True)
    # for image, label in train_loader:
    #     print('image shape:',image.shape)
    #     print('label shape:',label.shape)

    test_data()
    # pass
