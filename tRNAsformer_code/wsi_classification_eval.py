import torch
from tqdm import tqdm
from arguments import achieve_arguments
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
from utils.utils import load_pickle
import glob
import pandas as pd
from models.vit import VisionTransformer
from scipy import stats

def groupedAvg(myArray, N=2):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

def evaluate_model(model, args):
    for test in ['TCGA', 'external']:
        if test == 'TCGA':
            test_case_ids = load_pickle('splits/case_ids/test.txt')
            test_df = pd.read_csv('splits/image_gene_data/test.csv')
            instances = [row[1]['image_files'].replace('/instances/', '/20X_instances_spatially_ordered_fixed/') for row in test_df.iterrows()]
        else:
            test_case_ids = glob.glob('D:/external_data/external_slides/slides/*/*')
            instances = glob.glob('D:/external_data/external_slides/slides/*/*/20X_instances_spatially_ordered_fixed/*_instance_*.npy')
        
        instances = [f.replace('\\', '/') for f in instances]
        if test == 'TCGA':
            test_instances = [f for f in instances if f.split('/')[-3] in test_case_ids]
        else:
            test_instances = instances
        
        # legend_dict = {'Clear cell adenocarcinoma, NOS': 'ccRCC', 
        #                 'Renal cell carcinoma, chromophobe type': 'crRCC', 
        #                 'Papillary adenocarcinoma, NOS': 'pRCC', 
        #                'CCa Slides': 'ccRCC', 
        #                 'CH Ca Slides': 'crRCC', 
        #                 'Papillary Slides': 'pRCC'}
        
        if test == 'TCGA':
            label_dict = {'Clear cell adenocarcinoma, NOS': 0, 
                        'Renal cell carcinoma, chromophobe type': 1, 
                        'Papillary adenocarcinoma, NOS': 2}
            test_labels = [label_dict[f.split('/')[-4]] for f in test_instances]
            # case_ids = [f.split('/')[5] for f in test_instances]
            # legend_labels = [legend_dict[f.split('/')[-4]] for f in test_instances]
        else:
            label_dict = {'CCa Slides': 0, 
                        'CH Ca Slides': 1, 
                        'Papillary Slides': 2}
            test_labels = [label_dict[f.split('/')[-4]] for f in test_instances]
            # case_ids = [f.split('/')[5] for f in test_instances]
            # legend_labels = [legend_dict[f.split('/')[-4]] for f in test_instances]
        
        test_preds = []
        class_preds_all = []
        all_features = np.zeros((len(test_instances), args.embed_dim))
        for idx, instance in tqdm(enumerate(test_instances)):
            features = np.load(instance, allow_pickle=True)[()]['feature']
            vit_out = model.forward_features(torch.tensor(features).cuda().unsqueeze(0).unsqueeze(0))
            _, class_preds = model(torch.tensor(features).cuda().unsqueeze(0).unsqueeze(0))
            pred = class_preds.argmax().detach().cpu().numpy()[()]
            test_preds.append(pred)####################
            class_preds_all.append(class_preds.detach().cpu().numpy()[()])
            all_features[idx, ...] = vit_out[0, 0].detach().cpu().numpy()
            
        print('******************** Test: {}    ********************'.format(test))
        print('Accuracy: {:.4f}'.format(accuracy_score(test_labels, test_preds)))
        print('F1 score (micro): {:.4f}'.format(f1_score(test_labels, test_preds, average='micro')))
        print('F1 score (macro): {:.4f}'.format(f1_score(test_labels, test_preds, average='macro')))
        print('F1 score (weighted): {:.4f}'.format(f1_score(test_labels, test_preds, average='weighted')))
        print('Confusion matrix: \n{}'.format(confusion_matrix(test_labels, test_preds)))
        
        if test == 'TCGA':
            n_samples = 100
        else:
            n_samples = 100
        
        test_preds_np = np.asarray(test_preds).squeeze()
        class_preds_all_avg_pred = []
        for counter in range(0, test_preds_np.shape[0], n_samples):
            class_preds_all_avg_pred.append(stats.mode(test_preds_np[counter:counter+n_samples])[0][0])
        test_labels_avg = [t for t_idx, t in enumerate(test_labels) if (t_idx+1)%n_samples == 0]
        print('******************** Test (slide-level): {}    ********************'.format(test))
        print('Accuracy: {:.4f}'.format(accuracy_score(test_labels_avg, class_preds_all_avg_pred)))
        print('F1 score (micro): {:.4f}'.format(f1_score(test_labels_avg, class_preds_all_avg_pred, average='micro')))
        print('F1 score (macro): {:.4f}'.format(f1_score(test_labels_avg, class_preds_all_avg_pred, average='macro')))
        print('F1 score (weighted): {:.4f}'.format(f1_score(test_labels_avg, class_preds_all_avg_pred, average='weighted')))
        print('Confusion matrix: \n{}'.format(confusion_matrix(test_labels_avg, class_preds_all_avg_pred)))
        
    return

if __name__ == "__main__":
    args = achieve_arguments()

    device = 'cuda'

    if args.model == 'multitask':
        from models.vit import VisionTransformer
        model = VisionTransformer(img_size=args.tile_size, 
                                    patch_size=args.vit_patch_size, 
                                    in_chans=1, 
                                    n_classes=3, 
                                    embed_dim=args.embed_dim, 
                                    have_pos_embed=args.have_pos_embed, 
                                    depth=args.num_blocks, 
                                    n_heads=args.n_heads, 
                                    mlp_ratio=args.mlp_ratio, 
                                    attn_p=0.0)
        
        if args.experiment_name:
            experiment_name = "L{}_E{}_H{}_POS{}_O{}_{}".format(args.num_blocks, 
                                                        args.embed_dim, 
                                                        args.n_heads, 
                                                        int(args.have_pos_embed), 
                                                        args.optimizer, 
                                                        args.experiment_name)
        
        else:
            experiment_name = "L{}_E{}_H{}_POS{}_O{}".format(args.num_blocks, 
                                                    args.embed_dim, 
                                                    args.n_heads, 
                                                    int(args.have_pos_embed), 
                                                    args.optimizer)
    
    if args.model == 'ViT':
        from models.vit import VisionTransformer
        model = VisionTransformer(img_size=args.tile_size, 
                                    patch_size=args.vit_patch_size, 
                                    in_chans=1, 
                                    n_classes=3, 
                                    embed_dim=args.embed_dim, 
                                    have_pos_embed=args.have_pos_embed, 
                                    model_type=args.model, 
                                    depth=args.num_blocks, 
                                    n_heads=args.n_heads, 
                                    mlp_ratio=args.mlp_ratio, 
                                    attn_p=0.0)
        
        if args.experiment_name:
            experiment_name = "L{}_E{}_H{}_POS{}_O{}_{}_{}".format(args.num_blocks, 
                                                        args.embed_dim, 
                                                        args.n_heads, 
                                                        int(args.have_pos_embed), 
                                                        args.optimizer, 
                                                        args.model, 
                                                        args.experiment_name)
        
        else:
            experiment_name = "L{}_E{}_H{}_POS{}_O{}_{}".format(args.num_blocks, 
                                                    args.embed_dim, 
                                                    args.n_heads, 
                                                    int(args.have_pos_embed), 
                                                    args.optimizer, 
                                                    args.model)
    
    model = model.to(device)

    print('experiment_name: {}'.format(experiment_name))
    prefix = 'best_loss'

    MODEL_PATH = 'checkpoints/{0}/{0}_{1}.pt'.format(experiment_name, prefix)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    evaluate_model(model, args)