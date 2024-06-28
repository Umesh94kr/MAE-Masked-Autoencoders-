## importing necessay library

import torch
import torch.nn as nn
import timm       # for loading pytorch Image models
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indices(num_patches : int):
    # if num_patches = 256 the shuffled = [1 2 3 4 5 .... 255]
    shuffled = np.arange(num_patches)
    np.random.shuffle(shuffled)
    # this unshuffled would be used to sort the patches before sending to encoder
    unshuffled = np.argsort(shuffled)
    return shuffled, unshuffled


## shuffling patches class returns the randomly shuffled patches with backward indices 

class ShufflePatch(nn.Module):

    def __init__(self, mask_ratio)->None:
        super().__init__()
        self.ratio = mask_ratio

    def forward(self, patches : torch.Tensor):
        # if patches shape is 256, 128, 192 
        num_patches, batch, embd_dim = patches.shape

        # number of unmasked patches (visible patches) we want in an image
        # if ratio = 0.75 the number of unmasked patches = 192
        num_unmasked_patches = int((1 - self.ratio)*num_patches)

        # getting the random indices of the patches in an image
        indices = [random_indices(num_patches) for _ in range(batch)]
        
        # incdices would have a shape of (128, (256, 256))

        shuffled_indices = torch.as_tensor(np.stack([i[0] for i in indices], axis=-1), dtype=torch.long).to(patches.device)
        # shuffled indices shape = (256, 128)
        # unshuffled indices shape = (256, 128)
        unshuffle_indices = torch.as_tensor(np.stack([i[1] for i in indices], axis=-1), dtype=torch.long).to(patches.device)

        # shuffling patches
        patches = torch.gather(patches, 0, repeat(shuffled_indices, 'num_patches batch -> num_patches batch channels', channels=patches.shape[-1]))

        # selecting patches to be declared as unmasked
        patches = patches[:num_unmasked_patches]

        return patches, shuffled_indices, unshuffle_indices
    

## Masked Autoencoder Encodder

class MAE_encoder(nn.Module):
    def __init__(self, img_size=32, patch_size=2, embd_dim=198, num_layers=12, num_head=3, mask_ratio=0.75)->None:
        super().__init__()
        # During self-attention and other operations within the transformer, the class token interacts with all other tokens, effectively aggregating information from the entire sequence.

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embd_dim))

        # positional embeddings are used to inject information about the relative or absolute position of patches in the image
        self.pos_embeddings = torch.nn.Parameter(torch.zeros((img_size//patch_size)**2, 1, embd_dim))
        self.shuffle = ShufflePatch(mask_ratio)

        self.give_patches = torch.nn.Conv2d(3, embd_dim, patch_size, patch_size)  ## channels, embd_dim, filter, stride

        # transformer layer with multi-head self attention, feed-forward neural network and layer normalization
        self.transformer = torch.nn.Sequential(*[Block(embd_dim, num_head) for _ in range(num_layers)])
        self.layer_norm = torch.nn.LayerNorm(embd_dim)

        self.init_weights() # to properly initialize weights

    def init_weights(self):
        trunc_normal_(self.cls_token, std=0.2)
        trunc_normal_(self.pos_embeddings, std=0.2)

    def forward(self, img):

        # get patches from image
        patches = self.give_patches(img)

        # rearrange the dimensions
        patches = rearrange(patches, 'batch channels height width -> (height width) batch channels')
        patches += self.pos_embeddings

        # getting random viisble patches, shuffled indices, and unshuffled indices
        patches, shuffled_ind, unshuffled_ind = self.shuffle(patches)

        ## patches shape = (192, 128, 198)
        ## concatenating a class token to the sequence of patch embeddings shape = (193, 128, 198)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 'num_patches batch channels -> batch num_patches channels')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b n c -> n b c')

        return features, unshuffled_ind
    

## Maked Autoencoder Decoder

class MAE_decoder(nn.Module):

    def __init__(self, img_size=32, patch_size=2, embd_dim=198, num_layers=4, num_head=3, mask_ratio=0.75)->None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embd_dim))
        self.pos_embd = torch.nn.Parameter(torch.zeros((img_size//patch_size)**2 + 1, 1, embd_dim))

        self.transformer = torch.nn.Sequential(*[Block(embd_dim, num_head) for _ in range(num_layers)])

        self.head = torch.nn.Linear(embd_dim, 3 * patch_size**2) ## from 198 to 12
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=img_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embd, std=0.02)

    def forward(self, features, unshuffled_ind):
        num_patches = int(features.shape[0])

        zeros_tensor = torch.zeros(1, unshuffled_ind.shape[1]).to(unshuffled_ind)  # Shape: (1, 128)

        # Increment backward_indexes by 1
        incremented_indexes = unshuffled_ind + 1  # Shape: (256, 128)

        # Concatenate the tensors along the first dimension shape : (257, 128)
        unshuffled_indexes = torch.cat([zeros_tensor, incremented_indexes], dim=0) 

        ## concatenating masked tokens with unmasked ones shape : (257, 128, 198)
        masked_tokens = self.mask_token.expand(unshuffled_indexes.shape[0] - num_patches, features.shape[1], -1)
        features = torch.cat([features, masked_tokens], dim=0)

        ## unshuffle the total patchess to get aligned according to unshuffled indixes
        features = torch.gather(features, 0, repeat(unshuffled_indexes, 'num_patches batch -> num_patches batch channels', channels=features.shape[-1]))

        features += self.pos_embd

        features = rearrange(features, 'num_patches batch channels -> batch num_patches channels')
        features = self.transformer(features)
        features = rearrange(features, 'batch num_patches channels -> num_patches batch channels')
        ## need to make its shape (256, 128, 198) remove global feature
        features = features[1:]

        patches = self.head(features) # shape is (256, 128, 12)
        mask = torch.zeros_like(patches)
        mask[num_patches-1:] = 1



        mask = torch.gather(mask, 0, repeat(unshuffled_indexes[1:] - 1, 'num_patches batch -> num_patches batch channels', channels=mask.shape[-1]))

        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
    

## Main model MAE_Vit which contains encoder + decoder , used while pretraining

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=198,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, unshuffled_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  unshuffled_indexes)
        return predicted_img, mask
    



    



    










    





