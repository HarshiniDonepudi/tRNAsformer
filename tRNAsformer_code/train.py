# import random
import torch
from utils.dataset_utils import ImageGeneTransformerDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import copy
# from tqdm import tqdm
from arguments import achieve_arguments
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from utils.utils import make_directory, logging_func

def train_model(model, dataloaders, optimizer, scheduler, device, num_epochs, save_dir, log_dir, args):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {}\n'.format(pytorch_total_params))
    logging_func(log_dir, 'Total number of parameters: {}\n'.format(pytorch_total_params))
    
    since = time.time()

    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    final_model = copy.deepcopy(model)

    if args.model == 'multitask':
        best_loss = np.inf
        best_acc = 0.0

        criterion_1 = nn.MSELoss()
        criterion_2 = nn.CrossEntropyLoss()
    
    elif args.model == 'he2rna':
        best_loss = np.inf

        criterion_1 = nn.MSELoss()
    
    elif args.model == 'ViT':
        best_loss = np.inf
        best_acc = 0.0

        criterion_2 = nn.CrossEntropyLoss()
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging_func(log_dir, '\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
        logging_func(log_dir, '-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            if args.model == 'he2rna':
                running_loss = 0.0
            
            elif args.model == 'multitask' or args.model == 'ViT':
                running_loss = 0.0
                running_corrects = 0

            for sample_batched in dataloaders[phase]: # tqdm(dataloaders[phase]):

                images = sample_batched['image_feature_vector'].float().to(device)
                genes = sample_batched['gene_vector'].float().to(device)
                labels = sample_batched['label'].long().to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if args.model == 'multitask':
                        gene_preds, class_preds = model(images)
                        loss = 0.5 * criterion_1(gene_preds, genes) + criterion_2(class_preds, labels)
                        _, preds = torch.max(class_preds, 1)

                    elif args.model == 'he2rna':
                        gene_preds = model(images)
                        loss = criterion_1(gene_preds, genes)

                    elif args.model == 'ViT':
                        _, class_preds = model(images)
                        loss = criterion_2(class_preds, labels)
                        _, preds = torch.max(class_preds, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if args.model == 'multitask' or args.model == 'ViT':
                    running_corrects += torch.sum(preds == labels.data)
                    running_loss += loss.item() * labels.size(0)

                elif args.model == 'he2rna':
                    running_loss += loss.item() * labels.size(0)
            
            if args.model == 'multitask' or args.model == 'ViT':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))
                logging_func(log_dir, '\n{} Loss: {:.6f} Acc: {:.6f}\n'.format(phase, epoch_loss, epoch_acc))
            
            if args.model == 'he2rna':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                print('{} Loss: {:.6f}'.format(phase, epoch_loss))
                logging_func(log_dir, '\n{} Loss: {:.6f}\n'.format(phase, epoch_loss))

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), save_dir.format(epoch + 1))

            if phase == 'val' and best_loss > epoch_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_dir.format('best_loss'))
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)
                if args.model != 'he2rna':
                    if best_acc < epoch_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), save_dir.format('best_acc'))

        print()

    time_elapsed = time.time() - since
    if args.model == 'multitask' or args.model == 'ViT':
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val. accuracy: {:6f}'.format(best_acc))
        print('Best val. loss: {:6f}'.format(best_loss))
        logging_func(log_dir, 'Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        logging_func(log_dir, 'Best val. accuracy: {:6f}\n'.format(best_acc))
        logging_func(log_dir, 'Best val. loss: {:6f}\n'.format(best_loss))
    
    if args.model == 'he2rna':
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val. loss: {:6f}'.format(best_loss))
        logging_func(log_dir, 'Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        logging_func(log_dir, 'Best val. loss: {:6f}\n'.format(best_loss))

    final_model.load_state_dict(copy.deepcopy(model.state_dict()))
    model.load_state_dict(best_model_wts)
    return model, final_model, val_loss_history

if __name__ == "__main__":
    args = achieve_arguments()

    # seed = 0
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        model = HE2RNA(input_dim=1024, output_dim=31793, dropout=0.25, layers=args.he2rna_layers, ks=[1, 2, 5, 10, 20, 49])
        
        experiment_name = "HE2RNA_O{}".format(args.optimizer)
        for layer in args.he2rna_layers:
            experiment_name = experiment_name + '_' + str(layer)
    
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

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)#, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor=0.5)

    batch_size = args.batch_size

    train_image_gene_dataset = ImageGeneTransformerDataset(csv_file='splits/image_gene_data/train.csv')
    train_dataloader = DataLoader(train_image_gene_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_image_gene_dataset = ImageGeneTransformerDataset(csv_file='splits/image_gene_data/val.csv')
    val_dataloader = DataLoader(val_image_gene_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_image_gene_dataset = ImageGeneTransformerDataset(csv_file='splits/image_gene_data/test.csv')
    test_dataloader = DataLoader(test_image_gene_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    dataloaders_dict = {'train': train_dataloader, 
                        'val': val_dataloader, 
                        'test': test_dataloader}
    
    save_dir = os.path.join(args.saved_path, experiment_name, experiment_name + '_{}.pt').replace('\\', '/')
    make_directory(os.path.join(args.saved_path, experiment_name))
    log_dir = os.path.join(args.log_path, experiment_name, 'run.log').replace('\\', '/')
    make_directory(os.path.join(args.log_path, experiment_name))
    logging_func(log_dir, "Arguments:\n")
    for key in vars(args).keys():
        logging_func(log_dir, "{}: {}\n".format(key, vars(args)[key]))
    logging_func(log_dir, "########################\n")

    model, final_model, scratch_hist = train_model(model=model, 
                                                dataloaders=dataloaders_dict, 
                                                optimizer=optimizer, 
                                                scheduler=scheduler, 
                                                device=device, 
                                                num_epochs=args.n_epochs, 
                                                save_dir=save_dir, 
                                                log_dir=log_dir, 
                                                args=args)

    torch.save(model.state_dict(), save_dir.format('best_loss'))
    torch.save(final_model.state_dict(), save_dir.format('final_model'))