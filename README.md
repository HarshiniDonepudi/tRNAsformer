
# TRNAsformer

TRNAsformer is an attention-based topology that can learn both to predict the bulk RNA-seq from an image and represent the whole slide image of a glass slide simultaneously. The tRNAsformer uses multiple instance learning to solve a weakly supervised problem while the pixel-level annotation is not available for an image.


## Flow of system
![alt text](https://github.com/HarshiniDonepudi/tRNAsformers/blob/main/TRNAsformer_Architrcture.png)



## Flow of TRNAsformers:

Patches -\> Tissue Masks -\> K-Means clustering based on Histograms-\> Clusters = 49-\> Features extracted based on Densenet121 -\> 49\*1024-\>
                      Amirs approach(based on coordinates(special clustering)

Bags of instances created such that there are 7 rows and 7 columns-\> 224\*224 -\>      Random Sampling done based by taking samples from each 
                                                              with each instance 32\*32

cluster -\> .npy file generated for each instance created based on resampling -\> fed into vision Transformer and train.py file is run.

Dataset creation .csv file with image\_gene, gene\_expression,label (specifies the dataset where the image is taken,Kidney dataset that was used in the tRNAsformer paper had 3 different datasets)  is created both .npy format


The numpy array format for the instances should be in format:

Features can be read using;

features = np.load(instance, allow\_pickle=True)[()]['feature']

{'feature': array([[0.        , 1.9029806 , 0.        , ..., 0.01972643, 0.        ,
         0.55768365],
        [0.        , 0.01043449, 0.        , ..., 0.0267658 , 0.7615977 ,
         0.        ],
        [0.        , 0.8107864 , 0.01523623, ..., 0.02649011, 0.06823773,
         0.        ],
        ...,
        [0.45817223, 0.0175302 , 0.        , ..., 8.085226  , 0.04961037,
         2.0368629 ],
        [0.        , 0.        , 0.        , ..., 5.0472517 , 0.        ,
         0.        ],
        [0.        , 0.        , 0.03634823, ..., 0.14657755, 0.        ,
         0.05196447]], dtype=float32), 'locations': {20: array([[420,  476],
        [224,  560],
        [812,  420],
        [812,  560],
        [784,  728],
        [1260,  168],
        [504, 1064],
        [1176,  532],
        [1204,  896],
        [896, 1064],
        [1484,  112],
        [504, 1400],
        [1512,  868],
        [1568,  364],
        [924, 1400],
        [1092, 1344],
        [1288, 1232],
        [1876,   28],
        [532, 1652],
        [1904,  868],
        [1988,  336],
        [1736, 1260],
        [840, 2016],
        [2072,   84],
        [672, 2296],
        [1232, 1988],
        [2212, 1260],
        [2408,  560],
        [2268,  980],
        [924, 2212],
        [1400, 2324],
        [2240, 1596],
        [1792, 2184],
        [2436,  896],
        [2604,  644],
        [2100, 2100],
        [2436, 1120],
        [2408, 1680],
        [1764, 2324],
        [2940,  980],
        [2576, 1764],
        [3052, 1064],
        [2240, 2380],
        [2744, 1400],
        [2184, 2128],
        [2772, 1764],
        [2828, 2100],
        [2436, 2324],
        [2856, 2296]], dtype=int32)}}

Things that need to be done:

1. Resampling features from bag of instances to create instances.

- Criteria for Resampling: Randomly picking features from each cluster.

1. Matching the instances to FPKM data based on UUID.

Code Files:

Preprocess\_numpy.ipynb: All the detailed steps for preprocessing

Preprocess\_loop.py -\> code to propressing all the svs file and convert to .npy format



Problems;

Sometimes the code is giving different features after training using densenet121 though we mentioned that we need only 49 clusters.


## Usage

```javascript
# to train the model

python train.py 

# to evaluate gene prediction based on spearman and pearson correlation

python gene_pred_eval.py

# to evaluate wsi classification

python wsi_classification_eval.py
```


## Authors

- [@MehtabMurtaza](https://github.com/MehtabMurtaza)
- [@HarshiniDonepudi](https://github.com/MehtabMurtaza)
- [@NareshProdduturi](https://github.com/m081429)
- [@AmirSafarpoor](https://scholar.google.com/citations?user=_HBHGL4AAAAJ&hl=en)
- [@Hamid R Tizhoosh](https://www.mayo.edu/research/faculty/tizhoosh-hamid-r-ph-d/bio-20530617)
- [@Xiaojia Tang](https://scholar.google.com/citations?user=QI6LJSYAAAAJ&hl=en)
- [@Kevin J Thompson](http://kalarikrlab.org/Kevin.html)
- [@Krishna R Kalari](https://www.mayo.edu/research/faculty/kalari-krishna-r-ph-d/bio-00095546)



## References

 - [Code](https://www.nature.com/articles/s42003-023-04583-x#Sec6)
 - [Preprocessing](https://www.sciencedirect.com/science/article/pii/S1361841520301213e)
 - [HE2RNA](https://www.nature.com/articles/s41467-020-17678-4)


## Acknowledgements

- [Kalari Lab](http://kalarikrlab.org/Kalari.html)
 - [Kimia Lab](https://kimialab.uwaterloo.ca/kimia/)
 
