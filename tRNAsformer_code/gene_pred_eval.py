import torch
import numpy as np
# from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from utils.dataset_utils import ImageGeneTransformerDataset
from torch.utils.data import DataLoader
from statsmodels.stats.multitest import multipletests
from arguments import achieve_arguments

device = 'cuda'

n_genes = 31793
p_value = 0.01

args = achieve_arguments()
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
elif args.model == 'he2rna':
    from models.he2rna import HE2RNA

    model = HE2RNA(input_dim=1024, output_dim=n_genes, dropout=0.25, layers=args.he2rna_layers, ks=[1, 2, 5, 10, 20, 49])
    
    experiment_name = "HE2RNA_O{}".format(args.optimizer)
    for layer in args.he2rna_layers:
        experiment_name = experiment_name + '_' + str(layer)

print('experiment_name: {}'.format(experiment_name))
prefix = 'best_loss'

model = model.to(device)
MODEL_PATH = 'checkpoints/{0}/{0}_{1}.pt'.format(experiment_name, prefix)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

test_image_gene_dataset = ImageGeneTransformerDataset(csv_file='splits/image_gene_data/test.csv')
test_dataloader = DataLoader(test_image_gene_dataset, batch_size=1, shuffle=False)

n_samples = len(test_dataloader)
print('n_samples: {}'.format(n_samples))

all_genes_true = np.zeros((len(test_dataloader), n_genes))
all_genes_pred = np.zeros((len(test_dataloader), n_genes))

idx = 0
for sample_batched in test_dataloader: # tqdm(test_dataloader):
    images = sample_batched['image_feature_vector'].float().to(device)
    genes = sample_batched['gene_vector']
    labels = sample_batched['label'].long().to(device)
    
    if args.model == 'multitask':
        gene_preds, _ = model(images)
    
    elif args.model == 'he2rna':
        gene_preds = model(images)
    
    all_genes_pred[idx, ...] = gene_preds.squeeze().detach().cpu().numpy()
    all_genes_true[idx, ...] = genes
    idx += 1

pvalue_spearmanr = np.zeros(n_genes)
pvalue_pearsonr = np.zeros(n_genes)
pvalue_kendalltau = np.zeros(n_genes)

rho_spearmanr = np.zeros(n_genes)
rho_pearsonr = np.zeros(n_genes)
rho_kendalltau = np.zeros(n_genes)

mean_absolute_error = np.zeros(n_genes)
root_mean_squared_error = np.zeros(n_genes)
relative_root_mean_squared_error = np.zeros(n_genes)

mean_genes_train = np.load('additional_files/mean_genes_training.npy')

for i in range(n_genes): # tqdm(range(n_genes)):
    rho, pval = spearmanr(all_genes_pred[..., i], all_genes_true[..., i])
    rho_spearmanr[i] = rho
    pvalue_spearmanr[i] = pval

    rho, pval = pearsonr(all_genes_pred[..., i], all_genes_true[..., i])
    rho_pearsonr[i] = rho
    pvalue_pearsonr[i] = pval

    rho, pval = kendalltau(all_genes_pred[..., i], all_genes_true[..., i])
    rho_kendalltau[i] = rho
    pvalue_kendalltau[i] = pval

    dif_pred = (all_genes_pred[..., i] - all_genes_true[..., i])

    mean_absolute_error[i] = np.abs(dif_pred).sum() / n_samples
    root_mean_squared_error[i] = np.sqrt(np.power(dif_pred, 2).sum() / n_samples)
    relative_root_mean_squared_error[i] = np.sqrt(np.power(dif_pred, 2).sum() / np.power(mean_genes_train[i] - all_genes_true[..., i], 2).sum())

print('mean_absolute_error: {:.2f} \nroot_mean_squared_error: {:.2f} \nrelative_root_mean_squared_error: {:.2f}'.format(mean_absolute_error.mean(), 
                                                                                                    root_mean_squared_error.mean(), 
                                                                                                    relative_root_mean_squared_error.mean()))

print('std mean_absolute_error: {:.2f} \nstd root_mean_squared_error: {:.2f} \nstd relative_root_mean_squared_error: {:.2f}'.format(mean_absolute_error.std(), 
                                                                                                    root_mean_squared_error.std(), 
                                                                                                    relative_root_mean_squared_error.std()))

print('**************************************************************************')
reject, pvals_corrected, _, _ = multipletests(pvalue_spearmanr, method='holm-sidak')
print('spearmanr holm-sidak: ', len(pvals_corrected[pvals_corrected<p_value]))

reject, pvals_corrected, _, _ = multipletests(pvalue_spearmanr, method='fdr_bh')
print('spearmanr fdr_bh: ', len(pvals_corrected[pvals_corrected<p_value]))

mean = np.mean(rho_spearmanr)
std = np.std(rho_spearmanr)
print('spearmanr rho mean: {} std: {}'.format(mean, std))

np.save('logs/{}/pvalue_spearmanr_{}.npy'.format(experiment_name, prefix), pvalue_spearmanr)
np.save('logs/{}/rho_spearmanr_{}.npy'.format(experiment_name, prefix), rho_spearmanr)

print('**************************************************************************')
reject, pvals_corrected, _, _ = multipletests(pvalue_pearsonr, method='holm-sidak')
print('pearsonr holm-sidak: ', len(pvals_corrected[pvals_corrected<p_value]))

reject, pvals_corrected, _, _ = multipletests(pvalue_pearsonr, method='fdr_bh')
print('pearsonr fdr_bh: ', len(pvals_corrected[pvals_corrected<p_value]))

mean = np.mean(rho_pearsonr)
std = np.std(rho_pearsonr)
print('pearsonr rho mean: {} std: {}'.format(mean, std))

np.save('logs/{}/pvalue_pearsonr_{}.npy'.format(experiment_name, prefix), pvalue_pearsonr)
np.save('logs/{}/rho_pearsonr_{}.npy'.format(experiment_name, prefix), rho_pearsonr)

print('**************************************************************************')
reject, pvals_corrected, _, _ = multipletests(pvalue_kendalltau, method='holm-sidak')
print('kendalltau holm-sidak: ', len(pvals_corrected[pvals_corrected<p_value]))

reject, pvals_corrected, _, _ = multipletests(pvalue_kendalltau, method='fdr_bh')
print('kendalltau fdr_bh: ', len(pvals_corrected[pvals_corrected<p_value]))

mean = np.mean(rho_kendalltau)
std = np.std(rho_kendalltau)
print('kendalltau rho mean: {} std: {}'.format(mean, std))

np.save('logs/{}/pvalue_kendalltau_{}.npy'.format(experiment_name, prefix), pvalue_kendalltau)
np.save('logs/{}/rho_kendalltau_{}.npy'.format(experiment_name, prefix), rho_kendalltau)
print()
print()