from torch import nn


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_c=3,
                 embed_dim=768,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        # (b,c,h,w) -> (b,c,hw) -> (b,hw,c)
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

if __name__ == '__main__':
    import torch

    x = torch.randn(1,3,224,224)
    model = PatchEmbed()
    
    # 224 // 16 -> 14
    # 经过卷积，图片被分割成14x14个网格，通道数变为768
    # (1,3,224,224) -> (1,768,14,14)
    # (1,768,14,14) -> (1,14*14,768)
    print(model(x).size()) # torch.Size([1,196,768])
