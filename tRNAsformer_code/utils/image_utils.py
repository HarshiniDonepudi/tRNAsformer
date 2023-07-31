import glob
import numpy as np
from copy import deepcopy
# import cv2
import skimage
# from skimage.morphology import binary_closing, square
from utils import Filter
import os
from tqdm import tqdm
from sklearn.cluster import KMeans

def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,2)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,0] / (D) - 1.0
    cy = (OD[:,:,1]-OD[:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,2)
    cx = np.expand_dims(cx,2)
    cy = np.expand_dims(cy,2)
            
    X_HSD = np.concatenate((cx,cy,D),2)
    return X_HSD

def segment_tissue(thumbnail):
    thumbnail_np = deepcopy(np.asarray(thumbnail))

    not_red   = Filter.filter_red_pen(thumbnail_np)
    not_blue  = Filter.filter_blue_pen(thumbnail_np)
    not_green = Filter.filter_green_pen(thumbnail_np)
    thumbnail_without_marker  = not_green & not_blue & not_red
    thumbnail_np[thumbnail_without_marker == False] = (np.ones((1, 3), dtype="uint8") * 255)

    thumbnail_std = np.std(thumbnail_np, axis=2)
    background_mask = 1 - (thumbnail_std < 5) * 1
    background_mask = skimage.morphology.binary_closing(background_mask, skimage.morphology.square(3))
    thumbnail_np[background_mask == 0] = (np.ones((1, 3), dtype="uint8") * 255)
    
    thumbnail_HSD = RGB2HSD(thumbnail_np.astype('float32') / 255.)
    # kernel = np.ones((30, 30), np.float32) / 900
    # thumbnail_D = cv2.filter2D(thumbnail_HSD[..., 2], -1, kernel)
    thumbnail_D = skimage.filters.gaussian(thumbnail_HSD[..., 2], sigma=1) # TODO: check skimage gaussian instead of cv2 filter
    thumbnail_np[thumbnail_D < 0.05] = (np.ones((1, 3), dtype="uint8") * 255)
    
    mask = np.zeros(thumbnail_np.shape)
    mask[thumbnail_np < 255] = 1
    mask[thumbnail_np == 255] = 0
    return thumbnail_np, mask[..., 0]

def compress_slide(slide_paths):
    command = "C:\\Users\\asafarpo\\vips-dev-8.10\\bin\\vips.exe im_vips2tiff {} {}:jpeg:75,tile:224x224,pyramid,,,,8"
    for path in tqdm(slide_paths):
        os.system(command.format(path, path.replace('.svs', '.tiff')))

def sorted_cluster(x, model=None):
    if model == None:
        model = KMeans()
    model = sorted_cluster_centers_(model, x)
    model = sorted_labels_(model, x)
    return model

def sorted_cluster_centers_(model, x):
    model.fit(x)
    new_centroids = []
    magnitude = []
    for center in model.cluster_centers_:
        magnitude.append(np.sqrt(center.dot(center)))
    idx_argsort = np.argsort(magnitude)
    model.cluster_centers_ = model.cluster_centers_[idx_argsort]
    return model

def sorted_labels_(sorted_model, x):
    sorted_model.labels_ = sorted_model.predict(x)
    return sorted_model