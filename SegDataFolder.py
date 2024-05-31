import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.transforms as _transform
import torch
from osgeo import gdal
try:
    import transform as T
except:
    import data_utils.transform as T

traindir = "train"
testdir = "test"
imagedir = 'images'
labeldir = 'labels_0-1'

channel_list = [
                 '7_B1', '7_B2', '7_B3', '7_B4', '7_B5', '7_B6', '7_B7',
                 
                 '8_B1', '8_B2', '8_B3', '8_B4', '8_B5', '8_B6', '8_B7',
               
                 '9_B1', '9_B2', '9_B3', '9_B4', '9_B5', '9_B6', '9_B7'
]
# channel_list = ['5_1_C11', '5_2_C12real', '5_3_C12imag', '5_4_C22', '5_5_alpha', '5_6_anisotropy', '5_7_entropy',
#                 '6_1_C11', '6_2_C12real', '6_3_C12imag', '6_4_C22', '6_5_alpha', '6_6_anisotropy', '6_7_entropy',
#                 '7_1_C11', '7_2_C12real', '7_3_C12imag', '7_4_C22', '7_5_alpha', '7_6_anisotropy', '7_7_entropy',
#                 '7_B1', '7_B2', '7_B3', '7_B4', '7_B5', '7_B6', '7_B7',
#                 '8_1_C11', '8_2_C12real', '8_3_C12imag', '8_4_C22', '8_5_alpha', '8_6_anisotropy', '8_7_entropy'
#                 '8_B1', '8_B2', '8_B3', '8_B4', '8_B5', '8_B6', '8_B7',
#                 '9_1_C11', '9_2_C12real', '9_3_C12imag', '9_4_C22', '9_5_alpha', '9_6_anisotropy', '9_7_entropy',
#                 '9_B1', '9_B2', '9_B3', '9_B4', '9_B5', '9_B6', '9_B7']
                # '7_B1', '7_B2', '7_B3', '7_B4', '7_B5', '7_B6', '7_B7',
                #
                # '8_B1', '8_B2', '8_B3', '8_B4', '8_B5', '8_B6', '8_B7',
                #
                # '9_B1', '9_B2', '9_B3', '9_B4', '9_B5', '9_B6', '9_B7']
trainfile = r'C:\Users\ADMIN\Desktop\WML\S1L8\Data\train\labels_0-1'
testfile = r'C:\Users\ADMIN\Desktop\WML\S1L8\Data\test\labels_0-1'

# mean_train = np.array([
#                        0.0671, 0.0535, 0.0668, 0.0558, 0.2377, 0.1513, 0.0945,
#
#                        0.0440, 0.0352, 0.0548, 0.0413, 0.3372, 0.1378, 0.0699,
#
#                       0.0403, 0.0332, 0.0644, 0.0496, 0.3278, 0.1492, 0.0710 ])
#
# std_train = np.array([
#                      0.0220, 0.0201, 0.0233, 0.0277, 0.1086, 0.0729, 0.0618,
#
#                      0.0499, 0.0467, 0.0454, 0.0472, 0.1279, 0.0559, 0.0362,
#
#                       0.0152, 0.0157, 0.0223, 0.0233, 0.1070, 0.0483, 0.0288])
# mean_train = np.array([0.0786, 2.3698e-04, 1.4804e-04, 0.0157, 18.3926, 0.6406, 0.6603,
#                      0.0896, -9.4519e-05, 4.0846e-05, 0.0155, 15.1670, 0.7086, 0.5876,
#                      0.1340, -1.2123e-04, -1.7230e-04, 0.0236, 15.7570, 0.6951, 0.6050,
#                       0.0671, 0.0535, 0.0668, 0.0558, 0.2377, 0.1513, 0.0945,
#                     0.1103, 2.7427e-04, -1.9634e-04, 0.0263, 20.7217, 0.5914, 0.7205,
#                       0.0440, 0.0352, 0.0548, 0.0413, 0.3372, 0.1378, 0.0699,
#                       0.1448, -2.6393e-04, -8.6348e-04, 0.0396, 22.1342, 0.5638, 0.7485,
#                      0.0403, 0.0332, 0.0644, 0.0496, 0.3278, 0.1492, 0.0710 ])
mean_train = np.array([
                    
                      0.0671, 0.0535, 0.0668, 0.0558, 0.2377, 0.1513, 0.0945,
                    
                      0.0440, 0.0352, 0.0548, 0.0413, 0.3372, 0.1378, 0.0699,
                      
                     0.0403, 0.0332, 0.0644, 0.0496, 0.3278, 0.1492, 0.0710])
std_train = np.array([
                    0.0220, 0.0201, 0.0233, 0.0277, 0.1086, 0.0729, 0.0618,
                    
                    0.0499, 0.0467, 0.0454, 0.0472, 0.1279, 0.0559, 0.0362,
                     
                     0.0152, 0.0157, 0.0223, 0.0233, 0.1070, 0.0483, 0.0288
                        ])
mean_test = np.array([
   
                         0.0531,  0.0412, 0.0568, 0.0444, 0.2613, 0.1450, 0.0834,
                        
                        0.0335,  0.0258, 0.0479, 0.0350, 0.3449, 0.1336, 0.0641,
                      
                        0.0415, 0.0348, 0.0682, 0.0531, 0.3528, 0.1559, 0.0744
                        ])
std_test = np.array([
   
                     0.0162, 0.0153, 0.0190, 0.0233, 0.1212, 0.0628, 0.0498,
                     
                     0.0253, 0.0251, 0.0270, 0.0318, 0.1258, 0.0554, 0.0304,
                    
                     0.0279, 0.0274, 0.0331, 0.0336, 0.1124, 0.0554, 0.0337])

# std_train = np.array([0.1744, 0.0282, 0.0278, 0.0246, 5.8579, 0.1239, 0.1371,
#                      0.2088, 0.0368, 0.0347, 0.0289, 4.3721, 0.0880, 0.1125,
#                      0.2197, 0.0332, 0.0356, 0.0278, 4.3396, 0.0889, 0.1087,
#                     0.0220, 0.0201, 0.0233, 0.0277, 0.1086, 0.0729, 0.0618,
#                     0.2065, 0.0364, 0.0360, 0.0305, 4.5393, 0.0949, 0.0918,
#                     0.0499, 0.0467, 0.0454, 0.0472, 0.1279, 0.0559, 0.0362,
#                      0.3296, 0.0336, 0.0370, 0.0308, 4.3709, 0.0877, 0.0806,
#                      0.0152, 0.0157, 0.0223, 0.0233, 0.1070, 0.0483, 0.0288])

# mean_test = np.array([
#                      0.0531,  0.0412, 0.0568, 0.0444, 0.2613, 0.1450, 0.0834,
#
#                      0.0335,  0.0258, 0.0479, 0.0350, 0.3449, 0.1336, 0.0641,
#
#                      0.0415, 0.0348, 0.0682, 0.0531, 0.3528, 0.1559, 0.0744])



# mean_test = np.array([0.0925,  8.9272e-05,  -1.1848e-05,  0.0191, 19.7460, 0.6108, 0.6941,
#                      0.1030,  -2.5573e-04, -1.0535e-04,  0.0186, 16.2912, 0.6842, 0.6188,
#                     0.1298,  -2.1548e-04, -3.4439e-04,  0.0239, 16.2553, 0.6841, 0.6192,
#                      0.0531,  0.0412, 0.0568, 0.0444, 0.2613, 0.1450, 0.0834,
#                     0.1023,  5.1733e-05,  -3.5210e-04,  0.0251, 21.0769, 0.5832, 0.7280,
#                     0.0335,  0.0258, 0.0479, 0.0350, 0.3449, 0.1336, 0.0641,
#                    0.1386, -4.8210e-04, -0.0011, 0.0411, 23.4955, 0.5329, 0.7735,
#                     0.0415, 0.0348, 0.0682, 0.0531, 0.3528, 0.1559, 0.0744])

# std_test = np.array([0.0987, 0.0073, 0.0080, 0.0163, 5.5857, 0.1189, 0.1279,
#                      0.0977, 0.0090, 0.0107, 0.0158, 4.3721, 0.0874, 0.1065,
#                     0.0747, 0.0106, 0.0076, 0.0121, 4.2035, 0.0860, 0.1037,
#                      0.0162, 0.0153, 0.0190, 0.0233, 0.1212, 0.0628, 0.0498,
#                      0.0756, 0.0095, 0.0086, 0.0107, 4.7155, 0.0980, 0.0914,
#                      0.0253, 0.0251, 0.0270, 0.0318, 0.1258, 0.0554, 0.0304,
#                      0.0768, 0.0109, 0.0079, 0.0142, 4.9747, 0.1025, 0.0878,
#                      0.0279, 0.0274, 0.0331, 0.0336, 0.1124, 0.0554, 0.0337])

def get_idx(channels):
    assert channels in [7, 14, 21, 56,35,1]
    if channels == 21:
        return list(range(21))
    elif channels == 14:
        return list(range(14))
    elif channels == 2:
        return list(range(7))[-2:]
    elif channels == 56:
        return list(range(56))
    elif channels == 35:
        return list(range(35))
    elif channels == 7:
        return list(range(7))

def getTransform(train=True, channel_idx=[0,1,2,3]):
    if train:
        transform = T.Compose(
            [
                #T.RandomHorizontalFlip(), #试试删除与否能否提高精度
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean_train[channel_idx],std=std_train[channel_idx])
            ]
        )
    else:
            transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean_test[channel_idx],std=std_test[channel_idx])
        ])
    return transform

_transform_test = _transform.Compose([
            _transform.ToTensor(),
            _transform.Normalize(mean=mean_test, std=std_test)
        ])


class semData(Dataset):
    def __init__(self, train=True, root='./Data', channels =14 , transform=None, selftest_dir=None): #modified here!    channels = 35
        self.train = train
        self.root = root
        self.dir = traindir if self.train else testdir
        if selftest_dir is not None:
            self.dir = selftest_dir
        self.channels = channels
        self.c_idx = get_idx(self.channels)
        if selftest_dir is not None: #modified here!
            self.file = os.path.join(self.root,selftest_dir,'labels_0-1')
        else:
            self.file = trainfile if train else testfile

        self.img_dir = os.path.join(self.root, self.dir, imagedir)
        #print(self.img_dir)
        self.label_dir = os.path.join(self.root, self.dir, labeldir)


        if transform is not None:
            self.transform = transform
        else:
            self.transform = getTransform(self.train, self.c_idx)
        
        #self.data_list = pd.read_csv(self.file).values
        #print(self.data_list)
        imges_sets = os.listdir(self.file)
        imges_sets = np.expand_dims(imges_sets,axis=0)
        self.data_list = imges_sets.reshape(-1,1)
        print(self.data_list)
        print(len(self.data_list))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,index):
        L = []
        lbl_name = self.data_list[index][0] # 1081.png
        p = lbl_name.split('.')[0]          # 1081
        for k in self.c_idx:
            img_path = p + '.tif'  #'1_'+p+'.tif'   p + '.tif'
            img_path = os.path.join(self.img_dir, channel_list[k],img_path)
            #img = Image.open(img_path)
            img = gdal.Open(img_path)
            cols=img.RasterXSize#图像长度
            rows=img.RasterYSize#图像宽度
            img=img.ReadAsArray(0,0,cols,rows)
            img = np.expand_dims(np.array(img),axis=0)
            L.append(img)

        # image = cv2.imread(os.path.join(self.root,img_path), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        image = np.concatenate(L, axis=0)
        label = cv2.imread(os.path.join(self.label_dir, lbl_name), cv2.IMREAD_GRAYSCALE)

        if image.shape[1] != label.shape[0] or image.shape[2] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + p + " " + lbl_name + "\n"))
        image = np.transpose(image,(1,2,0))
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        return {
            'X':image,
            'Y':label,
            'path': lbl_name
        }
    
    def TestSetLoader(self,root='./Data/test',file='test'):
        l = pd.read_csv(os.path.join(root,file)).values
        for i in l:
            filename = i[0]
            path = os.path.join(root, filename)
            image = gdal.Open(path)
            cols = image.RasterXSize  # 图像长度
            rows = image.RasterYSize  # 图像宽度
            image = image.ReadAsArray(0, 0, cols, rows)
            image = _transform_test(image)
            yield filename,image

if __name__ == "__main__":
    trainset = semData(train=False, channels=56, transform=None, selftest_dir='test')
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        img = data['X']
        label = data['Y']
        path = data['path']
        print(img.size(),label.max())

    

