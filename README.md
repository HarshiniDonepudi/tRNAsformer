
# TRNAsformer

TRNAsformer is an attention-based topology that can learn both to predict the bulk RNA-seq from an image and represent the whole slide image of a glass slide simultaneously. The tRNAsformer uses multiple instance learning to solve a weakly supervised problem while the pixel-level annotation is not available for an image.


## Flow of system
![alt text](https://github.com/HarshiniDonepudi/tRNAsformers/blob/main/TRNAsformer_Architrcture.png)


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
 
