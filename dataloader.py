import os
from os.path import isdir, exists, abspath, join

import random
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import PIL
#import cv2
import random

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.95):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)
        # because file names are same for both images and labels

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            current += 1

            # todo: load images and labels

            augment_choice = np.random.randint(0,9)
            augment_choice = 2

            image_path = self.data_files[current-1]
            label_path = self.label_files[current-1]
            image = Image.open(image_path)
            label = Image.open(label_path)

            if(augment_choice == 1):
                image = image.convert('L')
                enhancer = PIL.ImageEnhance.Brightness(image)
                image = enhancer.enhance(2.0)
                #image = br(image)
            if (augment_choice == 3):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            if (augment_choice == 5):
                image = image.rotate(90)
                label = label.rotate(90)


            image = image.resize((572, 572))
            label = label.resize((572, 572))
            label = label.crop((92, 92, 480, 480))

            data_image = np.asarray(image)
            label_image = np.asarray(label)
            data_image = data_image / 255
            label_image = label_image / 255

            # transform = transforms.Compose([])

            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))