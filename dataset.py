import torch
from torch.utils.data import Dataset

import numpy as np
from os.path import join
from PIL import Image
import random

class CachingImagesDataset(Dataset):
    def __init__(self, labels, images_folder, label_columns,
                 transform=None, cache_prob=-1,
                 image_filename_column='image'):
        self.transform = transform
        
        self.cache_prob = cache_prob
        self.cache = dict()
        
        self.images_folder = images_folder
        self.image_filename_column = image_filename_column
        self.label_columns = label_columns
        
        self.labels = labels
        
    def process_new_item(self, index):
        row = self.labels.iloc[index]
        file_path = join(self.images_folder, row[self.image_filename_column])
        with open(file_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
#         image = cv2.imread(self.images_folder + row[self.image_filename_column])[:, :, ::-1]
        labels = row[self.label_columns]
        
        if self.transform is not None:
            image = self.transform(image)
        
        sample = {'image': image,
                  'target': np.argmax(labels.values)}
            
        return sample
    
    def process_cached_item(self, index):
        return self.cache.get(index, None)

        
    def __getitem__(self, index):
        result = None
        if random.random() < self.cache_prob:
            result = self.process_cached_item(index)
        
        if result is None:
            result = self.process_new_item(index)
            if self.cache_prob > 0:
                self.cache[index] = result
                
        return result
    
    def __len__(self):
        return len(self.labels)