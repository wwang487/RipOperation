import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#路径是自己电脑里所对应的路径
datapath = r'E:\Python\DeepLearning\Datasets\testdata'
txtpath = r'E:\Python\DeepLearning\Datasets\testdata\label.txt'

class MyDataset(Dataset):
    def __init__(self,txtpath):
        #创建一个list用来储存图片和标签信息
        imgs = []
        #打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        datainfo = open(txtpath,'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0],words[1]))

        self.imgs = imgs
	#返回数据集大小
    def __len__(self):
        return len(self.imgs)
	#打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(datapath+'\\'+pic)
        pic = transforms.ToTensor()(pic)
        return pic,label
#实例化对象
data = MyDataset(txtpath)
#将数据集导入DataLoader，进行shuffle以及选取batch_size
data_loader = DataLoader(data,batch_size=2,shuffle=True,num_workers=0)
#Windows里num_works只能为0，其他值会报错
