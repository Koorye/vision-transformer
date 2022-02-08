from torch import nn


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 n_heads=8,
                 qkv_bias=False,
                 attn_dropout=0.,
                 proj_dropout=0.):
        super(Attention, self).__init__()

        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -.5

        # 此处将Q、K、V三个矩阵拼接在一起
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        b, n, c = x.shape

        # +1为class token
        # (b,n_patches+1,emb_dim) -> (b,n_patches+1,3*emb_dim)
        # -> (b,n_patches+1,3,n_heads,emb_dim_per_head)
        # -> (3,b,n_heads,n_patches+1,emb_dim_per_head)
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.n_heads, c//self.n_heads)
               .permute(2, 0, 3, 1, 4)
               )

        # (b,n_heads,n_patches+1,emb_dim_per_head)
        q,k,v = qkv[0], qkv[1], qkv[2]

        # (b,n_heads,n_patches+1,emb_dim_per_head)
        # @ (b,n_heads,emb_dim_per_head,n_patches+1)
        # -> (b,n_heads,n_patches+1,n_patches+1)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # (b,n_heads,n_patches+1,n_patches+1)
        # @ (b,n_heads,n_patches+1,emb_dim_per_head)
        # -> (b,n_heads,n_patches+1,emb_dim_per_head)
        # -> (b,n_patches+1,n_heads,emb_dim_per_head)
        # -> (b,n_patches+1,emb_dim)
        x = (attn @ v).transpose(1,2).reshape(b,n,c)

        # (b,n_patches+1,emb_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

if __name__ == '__main__':
    import torch
    
    x = torch.randn(1,196+1,768)
    model = Attention(768)

    # Q,K,V -> (768,768)
    # query: x @ Q -> (197,768) @ (768,768) -> (197,768)
    # 同理key,value: (197,768)
    # head为8，均分给8个head
    # 每个head的query,key,value为(197,768/8) -> (197,96)
    # 每个head的attn: query @ key^T -> (197,96) @ (96,197) -> (197,197)
    # 每个head的attn @ value -> (197,197) @ (197,96) -> (197,96)
    # 拼接每个head -> (197,8*96) -> (197,768)
    # 合成后输出 (197,768) -> (197,768)
    print(model(x).size()) # torch.Size([1,197,768])