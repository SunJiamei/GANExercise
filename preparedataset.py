from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import os
from torchvision.transforms import transforms


#global
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
default_transformer = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
# Dataset
class Lsundaset(Dataset):
    datasetname = 'LSUN_bedroom' # number of training images 3002924

    def __init__(self, file_folder, transformer=None):
        '''

        :param file_folder: the path to the images of lsun training samples
        :param transformer: perform image transformation and normalization
        '''
        self.file_folder = file_folder
        if transformer == None:
            self.transformer = default_transformer
        else:
            self.transformer = transformer
        print(self.file_folder)
        self.file_list = os.listdir(self.file_folder)
        # print(len(self.file_list))
        # print(self.file_list[0])

    def __getitem__(self, i):
        img_filepath = os.path.join(self.file_folder, self.file_list[i])
        assert os.path.isfile(img_filepath)
        image_data = Image.open(img_filepath).convert('RGB')
        image_data = self.transformer(image_data)
        return image_data # (3, 64,64)

    def __len__(self):
        return len(self.file_list)



def main_test():
    file_folder = '../dataset/lsun/lsun-master/bedroom_train_64/'
    lsundata = Lsundaset(file_folder)
    train_loader = torch.utils.data.DataLoader(lsundata, batch_size=64, shuffle=True,
                                              num_workers=1, pin_memory=True, sampler=None)
    for i, img in enumerate(train_loader):
        print(img.size())
        break

if __name__ == '__main__':
    main_test()