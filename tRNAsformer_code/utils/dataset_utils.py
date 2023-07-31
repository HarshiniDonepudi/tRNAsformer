from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SimpleFeatureDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature_path = self.df.iloc[idx]["features"]
        feature = np.load(feature_path)
        
        label = np.asarray(self.df.iloc[idx]["labels"])
        
        sample = {'feature': torch.from_numpy(feature), 
                    'label': torch.from_numpy(label)}
        return sample

class ImageGeneTransformerDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx]["image_files"].replace('/instances/', '/20X_instances_spatially_ordered_fixed/')
        # img_name = img_name.replace("D:/", "")
        image_feature_vector = np.expand_dims(np.load(img_name, allow_pickle=True)[()]['feature'], axis=0)
        
        gene_name = self.df.iloc[idx]["gene_files"]
        # gene_name = gene_name.replace("D:/", "")
        gene_vector = np.load(gene_name).squeeze()
        
        label = np.asarray(self.df.iloc[idx]["labels"])
        
        sample = {'image_feature_vector': torch.from_numpy(image_feature_vector), 
                    'gene_vector': torch.from_numpy(gene_vector), 
                    'label': torch.from_numpy(label)}

        return sample

class ImageGeneParallelTransformerDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx]["image_files"]
        image_feature_vector = np.expand_dims(np.load(img_name, allow_pickle=True)[()]['feature'], axis=0)

        gene_name = self.df.iloc[idx]["gene_files"].replace('/fpkm/', '/2d_fpkm/')
        gene_vector = np.expand_dims(np.load(gene_name), axis=0)
        
        label = np.asarray(self.df.iloc[idx]["labels"])
        
        sample = {'image_feature_vector': torch.from_numpy(image_feature_vector), 
                    'gene_vector': torch.from_numpy(gene_vector), 
                    'label': torch.from_numpy(label)}

        return sample

class ImageGeneSurvivalDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx]["image_files"]
        image_feature_vector = np.expand_dims(np.load(img_name, allow_pickle=True)[()]['feature'], axis=0)

        gene_name = self.df.iloc[idx]["gene_files"].replace('/fpkm/', '/2d_fpkm/')
        gene_vector = np.expand_dims(np.load(gene_name), axis=0)
        
        survtime = np.asarray(self.df.iloc[idx]["survtime"])
        
        censor = np.asarray(self.df.iloc[idx]["censor"])
        
        label = np.asarray(self.df.iloc[idx]["labels"])

        sample = {'image_feature_vector': torch.from_numpy(image_feature_vector), 
                    'gene_vector': torch.from_numpy(gene_vector), 
                    'survtime': torch.from_numpy(survtime), 
                    'censor': torch.from_numpy(censor), 
                    'label': torch.from_numpy(label)}

        return sample

class ImageGeneDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx]["image_files"]
        image_feature_vector = np.expand_dims(np.load(img_name, allow_pickle=True)[()]['feature'], axis=0)

        gene_name = self.df.iloc[idx]["gene_files"].replace('/fpkm/', '/2d_fpkm/')
        gene_vector = np.expand_dims(np.load(gene_name), axis=0)
        
        gmm_labels = np.asarray(self.df.iloc[idx]["gene_labels"])
        
        label = np.asarray(self.df.iloc[idx]["labels"])

        sample = {'image_feature_vector': torch.from_numpy(image_feature_vector), 
                    'gene_vector': torch.from_numpy(gene_vector), 
                    'gmm_labels': torch.from_numpy(gmm_labels), 
                    'label': torch.from_numpy(label)}

        return sample