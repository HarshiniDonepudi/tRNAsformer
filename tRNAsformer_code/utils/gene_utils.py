from tqdm import tqdm
from glob import glob
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def load_fpkm(path, case_ids, df_transcriptome, label_dict):
    
    gene_expressions = []
    labels = []
    fpkm_paths = []
    
    for case_id in tqdm(case_ids):
        primary_diagnosis = df_transcriptome[df_transcriptome['cases.0.case_id']==case_id]['cases.0.diagnoses.0.primary_diagnosis'].tolist()[0]
        fpkms = glob(os.path.join(path, primary_diagnosis, case_id, "*FPKM-UQ.txt").replace('\\', '/'))
        for fpkm in fpkms:
            lines = open(fpkm, 'r').read().split('\n')
            gene_expression = [float(l.split('\t')[1]) for l in lines[:-1]]
            gene_expressions.append(gene_expression)
            labels.append(label_dict[primary_diagnosis])
            fpkm_paths.append(fpkm.replace('\\', '/'))

    return (np.asarray(gene_expressions), np.asarray(labels), fpkm_paths)

def feature_selection(data, num_runs=10, n_features=32, feature_importances_th=1e-5):
    X_train, y_train, _ = data
    important_features = np.zeros(X_train.shape[1])
    for _ in range(num_runs):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, verbose=1)
        clf.fit(X_train, y_train)
        important_features[np.where(clf.feature_importances_ > feature_importances_th)[0]] += 1
    return np.argsort(important_features)[::-1][:n_features] # sort in descending order and then select top n_features

