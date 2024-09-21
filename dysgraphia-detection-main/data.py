import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ConvertImageDtype, Pad, Resize, PILToTensor
from torchvision.transforms.functional import to_pil_image
import pandas as pd

from utils import create_simple_splits, create_multiple_splits, get_IAM_statistics, get_base_statistics, get_bhk_features
from path import *

class IAMDL(Dataset):

    def __init__(self, set : str, device):
        assert set == 'testset' or set == 'trainset' or set == 'validationset'
        self.set =  set
        self.path = IAM 
        self.set_samples = self.__get_set_samples()
        self.max_width, self.max_height = 10000,10000 #get_IAM_statistics()
        self.device = device
    
    def __len__(self):
        return len(self.set_samples)
    
    def __getitem__(self, index):
        return self.set_samples[index]
    
    def __get_set_samples(self):
        set_samples = []
        # for author in os.listdir(self.set):
        #     writings = os.path.join(self.set, author)
        #     for png in os.listdir(writings):
        #             set_samples.append(os.path.join(writings, png))
        # return set_samples
        f = open(self.path / 'sentences.txt')
        chars = set()
        for line in f:
            if not line or line[0]=='#':
                continue
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            fileNameSplit = lineSplit[0].split('-')
            fileName = str(self.path) + '/sentences/' + fileNameSplit[0] +"/"\
                       + fileNameSplit[0] + '-' + fileNameSplit[1] +"/"+ lineSplit[0] + '.png'
            
            gtText = lineSplit[9].strip(" ").replace("|", " ")
            
            chars = chars.union(set(list(gtText)))
            set_samples.append(fileName)
        
        train_folders = [x.strip("\n") for x in open(self.path/"LWRT/train.uttlist").readlines()]
        validation_folders = [x.strip("\n") for x in open(self.path/"LWRT/validation.uttlist").readlines()]
        test_folders = [x.strip("\n") for x in open(self.path/"LWRT/test.uttlist").readlines()]

        trainSamples = []
        validationSamples = []
        testSamples = []

        for i in range(0, len(set_samples)):
            file = set_samples[i].split("/")[-1][:-4].strip(" ")
            folder = "-".join(file.split("-")[:-2])
            if (folder in train_folders): 
                trainSamples.append(set_samples[i])
            elif folder in validation_folders:
                validationSamples.append(set_samples[i])
            elif folder in test_folders:
                testSamples.append(set_samples[i])
        self.charList = sorted(list(chars))
        
        if self.set=='testset':
            return testSamples
        elif self.set=='trainset':
            return trainSamples
        else:
            return validationSamples
    
    def get_triplet(self, sample):
        pos_aut = '/'.join(sample.split("/")[:-1])
        anc_img = sample.split("/")[-1]
        pos_img = random.choice([a for a in os.listdir(pos_aut)])
        while(pos_img == anc_img):
            pos_img = random.choice([a for a in os.listdir(pos_aut)])

        neg_aut = os.path.join(self.path/'sentences', random.choice([a for a in os.listdir(self.path / 'sentences')]))
        neg_aut = os.path.join(neg_aut, random.choice([a for a in os.listdir(neg_aut)]))
        while(pos_aut == neg_aut):
            neg_aut = os.path.join(self.path/'sentences', random.choice([a for a in os.listdir(self.path / 'sentences')]))
            neg_aut = os.path.join(neg_aut, random.choice([a for a in os.listdir(neg_aut)]))
        neg_img = random.choice([a for a in os.listdir(neg_aut)])

        anchor_img = Image.open(os.path.join(pos_aut, anc_img))
        anchor_w, anchor_h = anchor_img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - anchor_w, self.max_height - anchor_h), fill=1.),
            Resize((128, 1024))
        ])
        anchor = transform(anchor_img)

        positive_img = Image.open(os.path.join(pos_aut, pos_img))
        positive_w, positive_h = positive_img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - positive_w, self.max_height - positive_h), fill=1.),
            Resize((128, 1024))
        ])
        positive = transform(positive_img)

        negative_img = Image.open(os.path.join(neg_aut, neg_img))
        negative_w, negative_h = negative_img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - negative_w, self.max_height - negative_h), fill=1.),
            Resize((128, 1024))
        ])
        negative = transform(negative_img)

        return anchor, positive, negative
    
    def batch_triplets(self, samples):
        
        batch_size = len(samples)
        anchors = torch.empty(size=(batch_size, 1, 128, 1024))
        positives = torch.empty(size=(batch_size, 1, 128, 1024))
        negatives = torch.empty(size=(batch_size, 1, 128, 1024))
        
        for batch, sample in enumerate(samples):
            anchors[batch], positives[batch], negatives[batch] = self.get_triplet(sample)
        
        return anchors.to(self.device), positives.to(self.device), negatives.to(self.device)

class DysgraphiaDL(Dataset):

    def __init__(self, base :  str, set : str, device : str, use_csv : bool = False, bhk : str = 'binary', labels = 'certified', split : int = 0):
        assert base == 'children' or base == 'adults'
        assert set == 'train' or set == 'validation' or set == 'test'
        create_simple_splits(DYSG)

        self.BASE = DYSG
        self.set_samples = self.__set_samples()
        self.max_width, self.max_height = 1000,1000 #get_base_statistics(base)
        self.device = device
        self.use_csv = use_csv
        self.bhk = bhk
        self.labels = labels

        if base == 'children': 
            self.labels_csv = pd.read_csv(os.path.join(self.BASE,'labels.csv'), header=0, index_col=0, sep=";")
        else: 
            self.labels_csv = None

        if use_csv: 
            _, self.pen_features = get_bhk_features(bhk=bhk)
        else: 
            self.pen_features = 0
    
    def __len__(self):
        return len(self.set_samples)
    
    def __getitem__(self, index):
        img = Image.open(self.set_samples[index]).convert('L')
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - img.size[0], self.max_height - img.size[1]), fill=1.),
            Resize((192, 512))
        ])
        img = transform(img)
        

        if 'No_Dysgraphic' in self.set_samples[index].split("/")[1] : 
            label = torch.tensor(0)
        else: 
            label = torch.tensor(1)
        return img.to(self.device), label.to(self.device), torch.empty((1))
    
    def __set_samples(self):
        set_samples = []
        for dir in os.listdir(str(self.BASE)):
            sample=[str(self.BASE)+"/"+dir+"/"+filename for filename in os.listdir(str(self.BASE)+"/"+dir)]
            set_samples= set_samples+ sample

        return set_samples
    
    def get_binary_weights(self):
        counter = [0, 0]
        for sample in self.set_samples:
            author = sample.split("/")[-2]
            label = self.labels_csv.filter(like=self.labels.upper()).loc[author].values[0]
            if label == 0: counter[0] += 1
            else: counter[1] += 1
        print(f"Samples per class: {counter}")
        print(f"Values: {[min(counter) / counter[0], min(counter) / counter[1]]}")
        return torch.tensor([min(counter) / counter[0], min(counter) / counter[1]]).to(self.device)
