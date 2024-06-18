import torch
import torch.nn as nn
from config import *
from time_position_embed import TimePositionEmbedding
from dit_block import Dit_block

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, num_channels, hidden_states):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(input_channel = num_channels, out_channels = hidden_states, kernel_size = patch_size, stride = patch_size)

    def forward(self, hidden_states):
        r"""
        Perform patch embedding
        Args:
            hidden_states: torch.Tensor, [b, c, h, w]
        Output:
            [b, sequence_length, embedding_dim]
        """
        hidden_states = self.projection(hidden_states)  # [b, output_channel, patch_size, patch_size]
        hidden_states = hidden_states.flatten(2).transpose(1,2) # [b, patch_size*patch_size, output_channel]
        return hidden_states


class Dit(nn.Module):
    def __init__(self, image_size, patch_size, input_channel, embedding_size, dit_block_num, num_heads, label_num):
        super(Dit, self).__init__()

        self.patch_size = patch_size
        self.embedding_size = embedding_size

        # TODO: patch_size cannot be divisible by image_size
        self.num_patches = (image_size // patch_size) ** 2

        # patch_embedding
        self.patch_embedding = PatchEmbedding(patch_size = patch_size, num_channels = input_channel, hidden_states = embedding_size)
        self.position_embedding = nn.Parameter(torch.randn([1, self.num_patches, embedding_size]))

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
            self.dit_blocks.append(Dit_block(embedding_size, num_heads))

        # layer norm
        self.norm = nn.LayerNorm(embedding_size)

        # linear back to patch
        self.linear = nn.Linear(embedding_size, input_channel*patch_size**2)

        