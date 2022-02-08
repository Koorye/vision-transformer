from typing import OrderedDict
import torch
from functools import partial
from torch import nn

from models.patch_embed import PatchEmbed
from models.encoder_block import EncoderBlock


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 n_classes=1000,
                 embed_dim=768,
                 depth=12,
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 representation_size=None,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=None):
        super(VisionTransformer, self).__init__()

        self.n_classes = n_classes
        self.n_features = self.embed_dim = embed_dim
        self.n_tokens = 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, n_patches+self.n_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Dropout概率随深度递增
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]

        self.blocks = nn.Sequential(*[
            EncoderBlock(embed_dim, n_heads, mlp_ratio, qkv_bias,
                         dropout, attn_dropout, proj_dropout, norm_layer)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.has_logits = True
            self.n_features = representation_size
            self.per_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分类器
        self.head = nn.Linear(self.n_features, n_classes)

        # 初始化参数
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_dropout(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        # 分类只使用class_token
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        return x

if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    
    model = VisionTransformer(n_classes=10)
    print(model(x).size()) # torch.Size([1,10])
