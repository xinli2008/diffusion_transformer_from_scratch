import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    r"""
    Perform dit block shift and scale
    Args:
        x:      torch.tensor, [b, L, c]
        shift:  torch.tensor, [b, c]
        scale:  torch.tensor, [b, c]
    Return:
        torch.tensor, [b, L, c]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiHeadSelfAttention(nn.Module):
    r"""
        multi head self attention
    """
    def __init__(self, embedding_size, num_heads, qkv_bias = True):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_embedding_size = embedding_size // num_heads
        assert self.head_embedding_size * num_heads == embedding_size, \
            "embedding_size should be divisible by num_heads"
        
        self.w_q = nn.Linear(embedding_size, embedding_size, bias = qkv_bias)
        self.w_k = nn.Linear(embedding_size, embedding_size, bias = qkv_bias)
        self.w_v = nn.Linear(embedding_size, embedding_size, bias = qkv_bias)

    def forward(self, hidden_states):
        r"""
        Perform multihead self attention forward
        Args:
            hidden_states: torch.Tensor, [b, L, c]
        Return:
            torch.Tensor, [b, L, c]
        """
        # linear
        query = self.w_q(hidden_states)
        key = self.w_k(hidden_states)
        value = self.w_v(hidden_states)

        # view
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_embedding_size).permute(0,2,1,3).contiguous()
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_embedding_size).permute(0,2,1,3).contiguous()
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_embedding_size).permute(0,2,1,3).contiguous()

        # attention_scores
        attention_scores = torch.matmul(query, key.transpose(-1,-2)) * query.shape[-1] ** -0.5
        attention_scores = F.softmax(attention_scores, dim = -1)

        attention_out = torch.matmul(attention_scores, value)

        # return to original size
        attention_out = attention_out.view(attention_out.shape[0], attention_out.shape[2], -1)
        return attention_out

class Dit_block(nn.Module):
    def __init__(self, embedding_size, num_heads, mlp_ratio = 4):
        super(Dit_block, self).__init__()
        
        self.norm1 = nn.LayerNorm(embedding_size, eps = 1e-6)
        self.attn = MultiHeadSelfAttention(embedding_size, num_heads)
        self.norm2 = nn.LayerNorm(embedding_size, eps = 1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * mlp_ratio, bias = True),
            nn.SiLU(),
            nn.Linear(embedding_size * mlp_ratio, embedding_size, bias = True)
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size * 6, bias = True)
        )

    def forward(self, hidden_states, condition):
        r"""
        Perform dit block forward
        Args:
            hidden_states: torch.Tensor, [B, L, embedding_size]
            condition:     torch.Tensor, [B, embedding_size]
        Return:
            torch.Tensor, [B, L, embedding_size]
        """

        x = hidden_states

        condition = self.adaLN(condition)

        alpha1, alpha2, beta1, beta2, gamma1, gamma2 = torch.chunk(condition, chunks = 6, dim = 1) # [b, embedding_size]

        # norm1
        hidden_states = self.norm1(hidden_states)  # [B, L, embedding_size]
        
        # scale and shift
        hidden_states = modulate(hidden_states, beta1, gamma1)  # [B, L, embedding_size]

        # multi-head-self-attention
        hidden_states = self.attn(hidden_states)

        # scale with alpha1
        hidden_states = hidden_states * alpha1.unsqueeze(1)

        # resudiaul block
        hidden_states = x + hidden_states

        x1 = hidden_states
        
        # norm2
        hidden_states = self.norm2(hidden_states)
        
        # scale and shift
        hidden_states = modulate(hidden_states, beta2, gamma2)

        # mlp
        hidden_states = self.mlp(hidden_states)

        # scale
        hidden_states = hidden_states * alpha2.unsqueeze(1)

        # residual block
        hidden_states = hidden_states + x1

        return hidden_states
        

if __name__ == "__main__":
    dit_block = Dit_block(128, 4)
    hidden_states = torch.randn([2, 10, 128])
    condition = torch.randn([2, 128])
    out = dit_block(hidden_states, condition)
    print(out.shape)