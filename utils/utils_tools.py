import cv2,os,tqdm
import numpy as np



def Split_Picture(image_path,save_path,high_pixel,wide_pixel):
        
        image = cv2.imread(image_path)
        image
        image_name = name.split('.')[0]
        H,W,C = image.shape
        #图片的行
        raws = int(np.floor(H/(high_pixel)))
        #图片的列
        column = int(np.floor(W/(wide_pixel)))

        n_H = raws*high_pixel
        n_W = column*high_pixel

        h_remainder_half = (H-n_H)//2
        w_remainder_half = (W-n_W)//2

        cut_size = np.zeros((high_pixel, wide_pixel, 3))
        for i in range(raws):
            for j in range(column):
                cut_size = image[h_remainder_half+i*(high_pixel):h_remainder_half+(i+1)*(high_pixel),w_remainder_half+j*(wide_pixel):w_remainder_half+(j+1)*(wide_pixel),:]
                save_image_path = os.path.join(save_path,str(image_name)
                    +"_"+str(i+1)+"_"+str(j+1)+".jpg")#+name.split('.')[-1]
                cv2.imwrite(save_image_path,cut_size)

if __name__ =="__main__":
    i = r"D:\31890\Desktop\codefile\Utils\o2"
    o = r"D:\31890\Desktop\codefile\Utils\o2_xiao"
    Split_Picture(i,o,512,512)
