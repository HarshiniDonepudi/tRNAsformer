import argparse
from distutils.util import strtobool

# define the function to get the arguments...
def achieve_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--lr', type=float, default=3e-4, help='the learning rate of the network')
    parse.add_argument('--batch_size', type=int, default=64, help='the batch size')
    parse.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
    parse.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'], help='the optimizer type')
    parse.add_argument('--model', type=str, default='multitask', choices=['multitask', 'he2rna', 'ViT'], help='the model type')
    parse.add_argument('--he2rna_layers', nargs="+", type=int)
    ###############################################################################
    parse.add_argument('--vit_patch_size', type=int, default=32, help='the patch size')
    parse.add_argument('--num_blocks', type=int, default=1, help='number of encoder blocks')
    parse.add_argument('--embed_dim', type=int, default=192, help='size of embedding dimension')
    parse.add_argument('--n_heads', type=int, default=1, help='number of attention heads')
    parse.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio')
    parse.add_argument('--have_pos_embed', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='have positional embedding or not')
    ###############################################################################
    parse.add_argument('--tile_size', type=int, default=224, help='the tile size')
    parse.add_argument('--n_clusters', type=int, default=49, help='the number of clusters for k-means')
    parse.add_argument('--n_instances', type=int, default=100, help='the number of instances to create from a slide')
    parse.add_argument('--saved_path', type=str, default='checkpoints/', help='the path that save models')
    parse.add_argument('--log_path', type=str, default='logs/', help='the path that logs')
    parse.add_argument('--experiment_name', type=str, default='', help='the experiment name')
    parse.add_argument('--classifier', type=str, default="rf", help='the type of the classifier: nn or rf')
    parse.add_argument('--save_mask', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='segment and save tissue mask')
    parse.add_argument('--feature_selection', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='apply GBT for feature selection')
    parse.add_argument('--gene2matrix', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='save gene expression as a 2d matrix')

    args = parse.parse_args()

    return args

