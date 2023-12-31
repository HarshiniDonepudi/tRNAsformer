# -*- coding: utf-8 -*-
"""Preprocess_numpy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18GzRl4CXfL03fvHp6Nv5S87tFqkUJPgr
"""
import openslide, numpy as np
import cv2
from cv2 import filter2D

import numpy as np
from queue import Queue

import os
import glob
import shutil

def process_svs_file(svs_file_path, output_folder):

    while True:


        tcga_slide = svs_file_path
        slide = openslide.open_slide(tcga_slide)

        slide.get_thumbnail((300, 300))

        magnification = slide.properties['openslide.mpp-x']

        print(f"Magnification: {magnification}x")

        def RGB2HSD(X):
            eps = np.finfo(float).eps
            X[np.where(X==0.0)] = eps

            OD = -np.log(X / 1.0)
            D  = np.mean(OD,3)
            D[np.where(D==0.0)] = eps

            cx = OD[:,:,:,0] / (D) - 1.0
            cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)

            D = np.expand_dims(D,3)
            cx = np.expand_dims(cx,3)
            cy = np.expand_dims(cy,3)

            X_HSD = np.concatenate((D,cx,cy),3)
            return X_HSD


        def clean_thumbnail(thumbnail):
            thumbnail_arr = np.asarray(thumbnail)

            # writable thumbnail
            wthumbnail = np.zeros_like(thumbnail_arr)
            wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

            # Remove pen marking here
            # We are skipping this

            # This  section sets regoins with white spectrum as the backgroud regoin
            thumbnail_std = np.std(wthumbnail, axis=2)
            wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
            thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
            kernel = np.ones((30,30),np.float32)/900
            thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
            wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
            return wthumbnail

        # Commented out IPython magic to ensure Python compatibility.
        import matplotlib.pyplot as plt
        # %matplotlib inline

        thumbnail = slide.get_thumbnail((500, 500))
        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = (cthumbnail.mean(axis=2) != 255)*1.

        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(thumbnail)
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(tissue_mask, cmap='Greys_r')
        # plt.axis('off')

        objective_power = int(slide.properties['openslide.objective-power'])
        print(objective_power)

        w, h = slide.dimensions

        # at 20x its 1000x1000
        patch_size = (objective_power/20.)*1000
        patch_size

        mask_hratio = (tissue_mask.shape[0]/h)*patch_size
        mask_wratio = (tissue_mask.shape[1]/w)*patch_size

        # iterating over patches
        patches = []

        for i, hi in enumerate(range(0, h, int(patch_size) )):

            _patches = []
            for j, wi in enumerate(range(0, w, int(patch_size) )):

                # check if patch contains 70% tissue area
                mi = int(i*mask_hratio)
                mj = int(j*mask_wratio)

                patch_mask = tissue_mask[mi:mi+int(mask_hratio), mj:mj+int(mask_wratio)]

                tissue_coverage = np.count_nonzero(patch_mask)/patch_mask.size

                _patches.append({'loc': [i, j], 'wsi_loc': [int(hi), int(wi)], 'tissue_coverage': tissue_coverage})

            patches.append(_patches)

        patches

        import tqdm

        # for patch to be considered it should have this much tissue area
        tissue_threshold = 0.7

        flat_patches = np.ravel(patches)
        for patch in tqdm.tqdm(flat_patches):

            # ignore patches with less tissue coverage
            if patch['tissue_coverage'] < tissue_threshold:
                continue

            # this loc is at the objective power
            h, w = patch['wsi_loc']

            # we will go obe level lower, i.e. (objective power / 4)
            # we still need patches at 5x of size 250x250
            # this logic can be modified and may not work properly for images of lower objective power < 20 or greater than 40
            patch_size_5x = int(((objective_power / 4)/5)*250.)

            patch_region = slide.read_region((w, h), 1, (patch_size_5x, patch_size_5x)).convert('RGB')

            if patch_region.size[0] != 250:
                patch_region = patch_region.resize((250, 250))

            histogram = (np.array(patch_region)/255.).reshape((250*250, 3)).mean(axis=0)
            patch['rgb_histogram'] = histogram

        from sklearn.cluster import KMeans
        selected_patches_flags = [patch['tissue_coverage'] >= tissue_threshold for patch in flat_patches]
        selected_patches = flat_patches[selected_patches_flags]

        kmeans_clusters = 49
        kmeans = KMeans(n_clusters = kmeans_clusters)
        features = np.array([entry['rgb_histogram'] for entry in selected_patches])

        kmeans.fit(features)

        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('hsv', kmeans_clusters)

        patch_clusters = np.zeros(np.array(patches).shape+(3,))


        for patch, label in zip(selected_patches, kmeans.labels_):
            patch_clusters[patch['loc'][0], patch['loc'][1], :] = cmap(label)[:3]
            patch['cluster_lbl'] = label

        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(thumbnail)
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(patch_clusters)
        # plt.axis('off')

        # Another hyperparameter of Yottixel
        # Yottixel has been tested with 5, 10, and 15 with 15 performing most optimally
        percentage_selected = 15

        mosaic = []

        for i in range(kmeans_clusters):
            cluster_patches = selected_patches[kmeans.labels_ == i]
            n_selected = max(1, int(len(cluster_patches)*percentage_selected/100.))

            km = KMeans(n_clusters=n_selected)
            loc_features = [patch['wsi_loc'] for patch in cluster_patches]
            ds = km.fit_transform(loc_features)

            c_selected_idx = []
            for idx in range(n_selected):
                sorted_idx = np.argsort(ds[:, idx])

                for sidx in sorted_idx:
                    if sidx not in c_selected_idx:
                        c_selected_idx.append(sidx)
                        mosaic.append(cluster_patches[sidx])
                        break

        patch_clusters = np.zeros(np.array(patches).shape+(3,))

        for patch in selected_patches:
            patch_clusters[patch['loc'][0], patch['loc'][1], :] = np.array(cmap(patch['cluster_lbl'])[:3])*0.6
        for patch in mosaic:
            patch_clusters[patch['loc'][0], patch['loc'][1], :] = cmap(patch['cluster_lbl'])[:3]

        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(thumbnail)
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(patch_clusters)
        # plt.axis('off')

        import tensorflow as tf

        def preprocessing_fn(inp, sz=(1000, 1000)):

            out = tf.cast(inp, 'float') / 255.

            out = tf.cond(tf.equal(tf.shape(inp)[1], sz[0]),
                        lambda: out, lambda: tf.image.resize(out, sz))

            mean = tf.reshape((0.485, 0.456, 0.406), [1, 1, 1, 3])
            std = tf.reshape((0.229, 0.224, 0.225), [1, 1, 1, 3])

            out = out - mean
            out = out / std

            return out


        def get_dn121_model():
            model = tf.keras.applications.densenet.DenseNet121(input_shape=(1000, 1000, 3),\
                                                            include_top=False,\
                                                            pooling='avg')

            seq_model = tf.keras.models.Sequential([tf.keras.layers.Lambda(preprocessing_fn,\
                                                        input_shape=(None, None, 3),\
                                                        dtype=tf.uint8)])
            seq_model.add(model)
            return seq_model


        model = get_dn121_model()

        patch_queue = []
        feature_queue = []
        batch_size = 20

        for patch in tqdm.tqdm(mosaic):

            # this loc is at the objective power
            h, w = patch['wsi_loc']

            patch_size_20x = int((objective_power/20.)*1000)
            patch_region = slide.read_region((w, h), 0, (patch_size_20x, patch_size_20x)).convert('RGB')

            patch_queue.append(np.array(patch_region))
            if len(patch_queue) == batch_size:
                feature_queue.extend(model.predict( np.array(patch_queue) ))
                patch_queue = []

        if len(patch_queue) != 0:
            padded_arr = np.zeros((batch_size, patch_size_20x, patch_size_20x, 3))
            padded_arr[:len(patch_queue), :, :, :] = np.array(patch_queue)
            feature_queue.extend(model.predict( padded_arr )[:len(patch_queue)])

        features=np.array(feature_queue)
        print(features.shape)

        if features.shape == (49,1024):
            break


    # Define the dimensions
    num_instances = 49
    num_features = 1024
    instance_dim = 32

    random_features_queue=Queue()

    # Enqueue features into the queue (example)
    for _ in range(features.shape[0]):
        new_features = np.random.rand(num_features)
        random_features_queue.put(new_features)

    # Create a bag of instances
    bag_of_instances = np.zeros((features.shape[0], instance_dim, instance_dim))

    # Dequeue features and reshape into instances
    for i in range(num_instances):
        featuresusk = random_features_queue.get()
        instance = featuresusk.reshape((instance_dim, instance_dim))
        bag_of_instances[i] = instance

    # Verify the shape of the bag of instances
    print(bag_of_instances.shape)

    grid = bag_of_instances.reshape((7, 7, 32, 32))
    H_stacked = [np.hstack(row) for row in grid]
    final_array = np.vstack(H_stacked)
    print(final_array.shape)

    print(final_array)

    #converting into the specific format

    data_dict = {'feature': final_array}

    # Save the dictionary to a numpy file
    #np.save('test_1.npy', data_dict)

    output_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(svs_file_path))[0] + '.npy')
    np.save(output_file_path, data_dict)



def process_all_svs_files(root_directory, output_root_directory):
    # Get a list of all .svs file paths using the glob module
    svs_files = glob.glob(os.path.join(root_directory, '**', '*.svs'), recursive=True)

    for svs_file in svs_files:
        # Get the parent folder name (without the full path)
        folder_name = os.path.basename(os.path.dirname(svs_file))
        # Create a corresponding output folder inside the output_root_directory
        output_folder = os.path.join(output_root_directory, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Process the .svs file and save the .npy file inside the output_folder
        process_svs_file(svs_file, output_folder)

# Example usage:
#root_directory = '/Users/M295788/Documents/Mehtab/transformer_test/tRNAsformer/tRNAsformer/Preprocessing/Data_folder'  # Update this to the root directory containing your folders
#output_root_directory = '/Users/M295788/Documents/Mehtab/transformer_test/tRNAsformer/tRNAsformer/Preprocessing/Output_data'  # Update this to the root directory for the processed output

root_directory = '/research/bsi/projects/breast/s301449.LARdl/processing/harshini/he2RNAdata1'
output_root_directory = '/research/bsi/projects/breast/s301449.LARdl/processing/Mehtab_2/numpy_output'
process_all_svs_files(root_directory, output_root_directory)