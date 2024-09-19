import os
import random
from PIL import Image
import torch
import config
import numpy as np
import cv2
from tacobox import Taco
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ConvertImageDtype, Pad, Resize, PILToTensor
from torchvision.transforms.functional import to_pil_image
import pandas as pd

from utils import create_simple_splits, create_multiple_splits, get_IAM_statistics, get_base_statistics, get_bhk_features
from path import *

class Sample:
    "sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class IAMDL:
    def __init__(self, model_set, path, batch_size):
        assert model_set == 'test' or model_set == 'train' or model_set == 'validation'
        model_set += '.uttlist'
        self.path = path
        self.set = IAM / 'LWRT' / model_set
        self.batchSize = batch_size
        self.samples = []
        self.set_samples = []
        self.currIdx = 0
        self.charList = []
        self.max_width, self.max_height = get_IAM_statistics()
        
        # creating taco object for augmentation (checkout Easter2.0 paper)
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10
        )
        
        f = open(self.path + 'sentences.txt')
        chars = set()
        for line in f:
            if not line or line[0]=='#':
                continue
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            fileNameSplit = lineSplit[0].split('-')
            fileName = self.path + 'sentences/' + fileNameSplit[0] + '/' +\
                       fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
            
            gtText = lineSplit[9].strip(" ").replace("|", " ")
            
            chars = chars.union(set(list(gtText)))
            self.samples.append(fileName)
        
        folders = [x.strip("\n") for x in open(self.set).readlines()]


        for i in range(0, len(self.samples)):
            file = self.samples[i].split("/")[-1][:-4].strip(" ")
            folder = "-".join(file.split("-")[:-2])
            if (folder in folders): 
                self.set_samples.append(self.samples[i])
        # self.trainSet()
        self.charList = sorted(list(chars))
    
    def __len__(self):
        return len(self.set_samples)
    
    def __getitem__(self, index):
        return self.set_samples[index]
        
    # def trainSet(self):
    #     self.currIdx = 0
    #     random.shuffle(self.set_samples)
    #     self.samples = self.set_samples

    # def validationSet(self):
    #     self.currIdx = 0
    #     self.samples = self.set_samples
        
    # def testSet(self):
    #     self.currIdx = 0
    #     self.samples = self.set_samples
        
    def getIteratorInfo(self):
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        return self.currIdx + self.batchSize <= len(self.samples)
    
    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)
            
        # scaling image [0, 1]
        img = img/255
        img = img.swapaxes(-2,-1)[...,::-1]
        target = np.ones((config.INPUT_WIDTH, config.INPUT_HEIGHT))
        new_x = config.INPUT_WIDTH/img.shape[0]
        new_y = config.INPUT_HEIGHT/img.shape[1]
        min_xy = min(new_x, new_y)
        new_x = int(img.shape[0]*min_xy)
        new_y = int(img.shape[1]*min_xy)
        img2 = cv2.resize(img, (new_y,new_x))
        target[:new_x,:new_y] = img2
        return 1 - (target)
    
    def apply_taco_augmentations(self, input_img):
        random_value = random.random()
        if random_value <= config.TACO_AUGMENTAION_FRACTION:
            augmented_img = self.mytaco.apply_vertical_taco(
                input_img, 
                corruption_type='random'
            )
        else:
            augmented_img = input_img
        return augmented_img

    def getNext(self, what='train'):
        while True:
            if ((self.currIdx + self.batchSize) <= len(self.samples)):
                
                itr = self.getIteratorInfo()
                batchRange = range(self.currIdx, self.currIdx + self.batchSize)
                if config.LONG_LINES:
                    random_batch_range = random.choices(range(0, len(self.samples)), k=self.batchSize)
                    
                gtTexts = np.ones([self.batchSize, config.OUTPUT_SHAPE])
                input_length = np.ones((self.batchSize,1))*config.OUTPUT_SHAPE
                label_length = np.zeros((self.batchSize,1))
                imgs = np.ones([self.batchSize, config.INPUT_WIDTH, config.INPUT_HEIGHT])
                j = 0
                for ix, i in enumerate(batchRange):
                    img = cv2.imread(self.samples[i], cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
                    text = self.samples[i].gtText
                    
                    if config.LONG_LINES:
                        if random.random() <= config.LONG_LINES_FRACTION:
                            index = random_batch_range[ix]
                            img2 = cv2.imread(self.samples[index], cv2.IMREAD_GRAYSCALE)
                            if img2 is None:
                                img2 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
                            text2 = self.samples[index].gtText
                            
                            avg_w = (img.shape[1] + img2.shape[1])//2
                            avg_h = (img.shape[0] + img2.shape[0])//2
                            
                            resized1 = cv2.resize(img, (avg_w, avg_h))
                            resized2 = cv2.resize(img2, (avg_w, avg_h))
                            space_width = random.randint(config.INPUT_HEIGHT//4, 2*config.INPUT_HEIGHT)
                            space = np.ones((avg_h, space_width))*255
                            
                            img = np.hstack([resized1, space, resized2])
                            text = text + " " + text2
                            
                    if len(self.samples) < 3000:# FOR VALIDATION AND TEST SETS
                        eraser=-1
                    img = self.preprocess(img)                    
                    imgs[j] = img
                    
                    val = list(map(lambda x: self.charList.index(x), text))
                    while len(val) < config.OUTPUT_SHAPE:
                        val.append(len(self.charList))
                        
                    gtTexts[j] = (val)
                    label_length[j] = len(text)
                    input_length[j] = config.OUTPUT_SHAPE
                    j = j + 1
                    if False:
                        plt.figure( figsize = (20, 20))
                        plt.imshow(img)
                        plt.show()
                        
                self.currIdx += self.batchSize
                inputs = {
                        'the_input': imgs,
                        'the_labels': gtTexts,
                        'input_length': input_length,
                        'label_length': label_length,
                }
                outputs = {'ctc': np.zeros([self.batchSize])}
                yield (inputs,outputs)
            else:
                self.currIdx = 0
                
    def getValidationImage(self):
        batchRange = range(0, len(self.samples))
        imgs = []
        texts = []
        reals = []
        for i in batchRange:
            img1 = cv2.imread(self.samples[i], cv2.IMREAD_GRAYSCALE)
            real = cv2.imread(self.samples[i])
            if img1 is None:
                img1 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
            img = self.preprocess(img1, augment=False)
            img = np.expand_dims(img,  0)
            text = self.samples[i].gtText
            imgs.append(img)
            texts.append(text)
            reals.append(real)
        self.currIdx += self.batchSize
        return imgs,texts,reals
    
    def getTestImage(self):
        batchRange = range(0, len(self.samples))
        imgs = []
        texts = []
        reals = []
        for i in batchRange:
            img1 = cv2.imread(self.samples[i], cv2.IMREAD_GRAYSCALE)
            real = cv2.imread(self.samples[i])
            if img1 is None:
                img1 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
            img = self.preprocess(img1, augment=False)
            img = np.expand_dims(img,  0)
            text = self.samples[i].gtText
            imgs.append(img)
            texts.append(text)
            reals.append(real)
        self.currIdx += self.batchSize
        return imgs,texts,reals

# class IAMDL(Dataset):

#     def __init__(self, set : str, device):
#         assert set == 'test' or set == 'train' or set == 'validation'
#         self.set = IAM / 'LWRT' / set + '.uttlist'
#         self.set_samples = self.__get_set_samples()
#         self.max_width, self.max_height = get_IAM_statistics()
#         self.device = device
    
#     def __len__(self):
#         return len(self.set_samples)
    
#     def __getitem__(self, index):
#         return self.set_samples[index]
    
#     def __get_set_samples(self):
#         set_samples = []
#         for author in os.listdir(self.set):
#             writings = os.path.join(self.set, author)
#             for png in os.listdir(writings):
#                     set_samples.append(os.path.join(writings, png))
#         return set_samples
    
    def get_triplet(self, sample):
        pos_aut = '/'.join(sample.split("/")[:-1])
        anc_img = sample.split("/")[-1]
        pos_img = random.choice([a for a in os.listdir(pos_aut)])
        while(pos_img == anc_img):
            pos_img = random.choice([a for a in os.listdir(pos_aut)])

        neg_aut = os.path.join(self.path + 'sentences', random.choice([a for a in os.listdir(self.path + 'sentences')]))
        neg_aut = os.path.join(neg_aut, random.choice([a for a in os.listdir(neg_aut)]))
        while(pos_aut == neg_aut):
            neg_aut = os.path.join(self.path + 'sentences', random.choice([a for a in os.listdir(self.path + 'sentences')]))
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
        
        return anchors.to('cpu'), positives.to('cpu'), negatives.to('cpu')

class DysgraphiaDL(Dataset):

    def __init__(self, base :  str, set : str, device : str, use_csv : bool = False, bhk : str = 'binary', labels = 'certified', split : int = 0):
        assert base == 'children' or base == 'adults'
        assert set == 'train' or set == 'validation' or set == 'test'
        if base == 'children': create_multiple_splits(os.path.join(DYSG, base), os.path.join(DYSG, base, 'labels.csv'))
        else: create_simple_splits(os.path.join(DYSG, base))

        self.BASE = os.path.join(DYSG,base)
        self.SET = os.path.join(self.BASE,f"splits/{labels.upper()}/split{split}/{set}.txt")
        self.set_samples = self.__set_samples()
        self.max_width, self.max_height = get_base_statistics(base)
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
        aut_name = self.set_samples[index].split("/")[-2]
        img = Image.open(self.set_samples[index]).convert('L')
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, self.max_width - img.size[0], self.max_height - img.size[1]), fill=1.),
            Resize((192, 512))
        ])
        img = transform(img)
        
        if self.labels_csv is None:
            if 'O' in self.set_samples[index].split("/")[-2]: label = torch.tensor(0)
            else: label = torch.tensor(1)
        else:
            label = self.labels_csv.filter(like=self.labels.upper()).loc[aut_name].values[0]
            label = torch.tensor(label)

        if self.use_csv:
            pen_features, _ = get_bhk_features(self.set_samples[index], self.BASE.split("/")[-1], self.bhk)
            return img.to(self.device), label.to(self.device), pen_features.to(self.device)
        else:
            return img.to(self.device), label.to(self.device), torch.empty((1))
    
    def __set_samples(self):
        set_samples = []
        set_authors = [line.rstrip('\n') for line in open(self.SET, 'r')]
        AUTHORS = os.path.join(self.BASE, 'original')
        for author in os.listdir(AUTHORS):
            if author not in set_authors: continue
            LINES = os.path.join(AUTHORS, author)
            for png in os.listdir(LINES):
                set_samples.append(os.path.join(LINES, png))

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
