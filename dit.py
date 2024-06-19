import torch
import torch.nn as nn
from config import *
from time_position_embed import TimePositionEmbedding
from dit_block import Dit_block

class PatchEmbedding(nn.Module):
    r"""
    2D-image patch embedding
    """
    def __init__(self, patch_size, num_channels, hidden_states):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels = num_channels, out_channels = hidden_states, kernel_size = patch_size, stride = patch_size)

    def forward(self, hidden_states):
        r"""
        Perform patch embedding
        Args:
            hidden_states: torch.Tensor, [b, c, h, w]
        Output:
            [b, sequence_length, embedding_dim]
        """
        hidden_states = self.projection(hidden_states)  # [b, output_channel, patch_size, patch_size]
        hidden_states = hidden_states.flatten(2).transpose(1,2) # [b, patch_count*patch_count, output_channel]
        return hidden_states


class Dit(nn.Module):
    r"""
        stable diffusion with a transformer block
    """
    def __init__(self, image_size, patch_size, input_channel, embedding_size, dit_block_num, num_heads, label_num, mlp_ratio = 4):
        super(Dit, self).__init__()

        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.channels = input_channel

        # TODO: patch_size cannot be divisible by image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_count = image_size // self.patch_size

        # patch_embedding
        self.patch_embedding = PatchEmbedding(patch_size = patch_size, num_channels = input_channel, hidden_states = embedding_size)
        self.position_embedding = nn.Parameter(torch.randn([1, self.num_patches, embedding_size]).to(device))

        # time_embedding
        self.time_embedding = nn.Sequential(
            TimePositionEmbedding(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        # cls_embedding
        self.cls_embedding = nn.Embedding(num_embeddings = label_num, embedding_dim = embedding_size)

        # dit_block 
        self.dit_blocks = nn.ModuleList()
        for _ in range(dit_block_num):
            self.dit_blocks.append(Dit_block(embedding_size, num_heads, mlp_ratio))

        # layer norm
        self.norm = nn.LayerNorm(embedding_size)

        # linear back to patch
        self.linear = nn.Linear(embedding_size, input_channel * patch_size ** 2)

    def unpatchify(self, batch_x):
        r"""
        Perform diffusion transformer unpatchify
        Args:
            batch_x : [b, patch_count*patch_count, input_channel*patch_size**2]
        Output:
            torch.tensor, [b, c, h, w]
        """
        batch_x = batch_x.reshape(batch_x.shape[0], self.patch_count, self.patch_count, self.channels, self.patch_size, self.patch_size)
        batch_x = torch.permute(batch_x, dims=[0,3,1,2,4,5]).contiguous()  # [b,c,patch_count,patch_count,patch_size,patch_size]
        batch_x = torch.permute(batch_x, dims=[0,1,2,4,3,5]).contiguous()  # [b,c,patch_count,patch_size,patch_count,patch_size]
        batch_x = torch.reshape(batch_x, shape = [batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]*batch_x.shape[3], batch_x.shape[4]*batch_x.shape[5]])
        return batch_x

    def forward(self, batch_x, batch_t, batch_cls):
        r"""
        Perform diffusion transformer forward
        Args:
            batch_x: torch.Tensor, [b, c, h, w]
            batch_t: torch.Tneosr, [b, ]
            batch_t: torch.Tensor, [b, ]
        Return:
            torch.Tensor, [b, c, h, w]
        """
        batch_x = batch_x.to(device)
        batch_t = batch_t.to(device)
        batch_cls = batch_cls.to(device)

        # cls_embedding
        batch_cls = self.cls_embedding(batch_cls)

        # time_embedding
        batch_t = self.time_embedding(batch_t)

        # time_embedding + cls_embedding
        embedding = batch_t + batch_cls

        # patchify and add position embedding
        batch_x = self.patch_embedding(batch_x)  
        batch_x = batch_x + self.position_embedding  # [b, patch_count*patch_count, embedding_size]

        # dit block
        for layer in self.dit_blocks:
            batch_x = layer(batch_x, embedding)      #  [b, patch_count*patch_count, embedding_size]
        
        # layer norm
        batch_x = self.norm(batch_x)  # [b, patch_count*patch_count, embedding_size]

        # linear
        batch_x = self.linear(batch_x)  # [b, patch_count*patch_count, input_channel*patch_size**2]

        # unpatchify
        batch_x = self.unpatchify(batch_x)  # [b, c, h, w]

        return batch_x


if __name__ == "__main__":
    # init
    my_dit = Dit(image_size = 48, patch_size = 4, input_channel = 1, embedding_size = 128, dit_block_num = 4, num_heads = 4, label_num = 10, mlp_ratio = 4)
    my_dit = my_dit.to(device)

    x=torch.rand(2,1,48,48).to(device)
    t=torch.randint(0,timestep,(2,)).to(device)
    y=torch.randint(0,10,(2,)).to(device)

    out = my_dit(x, t, y)

    print(out.shape)
