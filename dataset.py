import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.utils import shuffle
import numpy as np
import pandas as pd

from os.path import join
from PIL import Image
import random


TARGET_LABEL_NAMES = ['MEL', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'NV']


def prepare_data_loaders(hparams):
    """Converts given hyperparameters to a pair of loaders.
       Arguments:
           hparams: dict of hyperparameters
       Returns:
           train_loader, valid_loader: dataloaders for training and validation
    """
    
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')
    
    if 'data_params' not in hparams:
        raise Exception('You must provide data params in hparams')
    
    data_params = hparams['data_params']
    if 'transforms' not in data_params or not isinstance(data_params['transforms'], list):
        raise Exception('You must add transforms list into hparams.data_params')
    
    transforms_list = []
    for transform_info in data_params['transforms']:
        transform_name = transform_info['name']
        transform_params = transform_info['params']
        if transform_params is not None:
            transform = transforms.__dict__[transform_name](**transform_params)
        else:
            transform = transforms.__dict__[transform_name]()
        transforms_list.append(transform)
    transform = transforms.Compose(transforms_list)
        
    if 'labels_path' not in data_params:
        raise Exception('You must add labels_path into hparams')
    
    if 'n_folds' not in data_params or 'train_folds' not in data_params or \
        'folds_split_column' not in data_params or 'folds_seed' not in data_params:
        raise Exception('You must add n_folds, train_folds, folds_split_column' \
                        'and folds_seed into hparams.data_params')
    
    labels = read_labels(data_params['labels_path'])
    labels = column_fold_split(labels, data_params['folds_split_column'],
                      data_params['folds_seed'], data_params['n_folds'])
    
    if 'images_path' not in data_params:
        raise Exception('You must add images_path into hparams.data_params')
    
    train_folds = list(map(int, data_params['train_folds'].split(",")))
    train_labels = labels[labels['fold'].isin(train_folds)]
    valid_labels = labels[~labels['fold'].isin(train_folds)]
    
    train_labels = train_labels.reset_index().drop('index', axis=1)
    valid_labels = valid_labels.reset_index().drop('index', axis=1)
    
    train_dataset = CachingImagesDataset(train_labels, data_params['images_path'],
                                         TARGET_LABEL_NAMES, transform=transform,
                                         image_filename_column=data_params['folds_split_column'])
    valid_transform = transforms.Compose(list(filter(
        lambda t: isinstance(t, transforms.Resize) or \
            isinstance(t, transforms.ToTensor) or \
            isinstance(t, transforms.Normalize), transforms_list)))
    valid_dataset = CachingImagesDataset(valid_labels, data_params['images_path'],
                                         TARGET_LABEL_NAMES, transform=valid_transform,
                                         image_filename_column=data_params['folds_split_column'])
    
    if 'training_params' not in hparams or 'batch_size' not in hparams['training_params']:
        raise Exception('You must add training_params with batch_size specified in hparams')
    training_params = hparams['training_params']
    
    n_workers = data_params['n_workers'] if 'n_workers' in data_params else 0
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'],
                              shuffle=True, num_workers=n_workers,
                              pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_dataset, batch_size=training_params['batch_size'],
                              shuffle=False, num_workers=n_workers,
                              pin_memory=torch.cuda.is_available())
    
    return train_loader, valid_loader


def column_fold_split(df, column, folds_seed, n_folds):
    df_tmp = []
    labels = shuffle(sorted(df[column].unique()), random_state=folds_seed)
    for i, fold_labels in enumerate(np.array_split(labels, n_folds)):
        df_label = df[df[column].isin(fold_labels)]
        df_label['fold'] = i
        df_tmp.append(df_label)
    df = pd.concat(df_tmp)
    
    return df


def read_labels(path):
    """Reads labels of the dataset from path.
    Arguments:
        path: str, path to csv file with labels
    Returns:
        labels: Pandas DataFrame with filenames and labels
    """
    labels = pd.read_csv(path,
                        dtype={**{'image': str},
                               **{label: int for label in TARGET_LABEL_NAMES}})
    labels.image += '.jpg'
    
    return labels


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