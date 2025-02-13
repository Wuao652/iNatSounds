from functools import partial

import torch
import torch.nn as nn

import timm
import timm.models.vision_transformer
from timm.models.vision_transformer import default_cfgs

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == "__main__":
    print("Hello world from models_vit.py !")
    vit_b_ckpt_path = "../mae_pretrain_weights/mae_pretrain_vit_base.pth"

    # define the vit_b model
    vit_model = vit_base_patch16(num_classes=5567)
    # print(vit_model)


    all_vit_models = timm.list_models('*vit*')
    print(all_vit_models)
    print("num of all vit models: ", len(all_vit_models))

    # 'vit_base_patch16_224.orig_in21k'
    # 'vit_base_patch16_224'
    vit_b_default_cfg = default_cfgs['vit_base_patch16_224']
    print(vit_b_default_cfg['url'])
    state_dict = torch.hub.load_state_dict_from_url(vit_b_default_cfg['url'], map_location='cpu')
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
    vit_model.load_state_dict(filtered_dict, strict=False)
    print("Loaded imagenet pretrained weights ... ")

    # load mae_pretrain model weights
    ckpt_weights = torch.load(vit_b_ckpt_path, map_location='cpu')["model"]
    filtered_weights = {k: v for k, v in ckpt_weights.items() if not k.startswith("head")}
    # print(filtered_weights.keys())
    print("Loaded mae pretrained weights ... ")


    import sys
    sys.exit(0)
