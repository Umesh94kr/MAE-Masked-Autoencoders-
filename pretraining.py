import math
import os

import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# importing all classes from model.py file
from model import *
from utils import setup_seed

if __name__ == '__main__':
    ## Parameters
    SEED = 42
    BATCH_SIZE = 50
    LEARNING_RATE = 1.5e-4
    WEIGHT_DECAY = 0.05
    MASK_RATIO = 0.75
    EPOCHS = 200
    MODEL_PATH = 'mae-autoencoder.pt'

    # setting up seed for reproducibilty of results
    setup_seed(SEED)

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])

    ## Dataset and DataLoaders
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    ## CIFAR contains 60,000 images , train_data have 50,000 and val_data have 10,000
    ## I only used 10,000 images to train

    train_dataset = Subset(train_dataset, range(1000))

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))

    ## setting up device to "cuda" if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # specifying the model
    model = MAE_ViT(mask_ratio=MASK_RATIO).to(device)

    # specifying optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE*4096/256, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    # learning rate scheduler
    lr_func = lambda epoch: min((epoch + 1) / (EPOCHS + 1e-8), 0.5 * (math.cos(EPOCHS / 2000 * math.pi) + 1))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)


    ## training loop
    step_count = 0

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)

            loss = torch.mean((predicted_img - img) ** 2 * mask)/MASK_RATIO
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
        
        lr_scheduler.step()
        avg_loss = sum(losses)/len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=epoch)
        print(f'In epoch {epoch}, average traning loss is {avg_loss}.')

        ## validation
        ## looking at first 5 predicted images of val dataset
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)

            predicted_val_img, mask = model(val_img)
            ## creating the predicted image
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)

            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)

            ## logging the images
            writer.add_image('mae_image', (img + 1) / 2, global_step=epoch)

        """saving the model"""
        torch.save(model, MODEL_PATH)

    writer.close()