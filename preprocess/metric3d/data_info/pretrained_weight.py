
mldb_info={}

mldb_info['checkpoint']={
    'mldb_root': '/data0/jxhuang/models', # NOTE: modify it to the pretrained ckpt root

    # pretrained weight for convnext
    'convnext_tiny': 'convnext/convnext_tiny_22k_1k_384.pth',
    'convnext_small': 'convnext/convnext_small_22k_1k_384.pth',
    'convnext_base': 'convnext/convnext_base_22k_1k_384.pth',
    'convnext_large': 'convnext/convnext_large_22k_1k_384.pth',
    'vit_large': 'vit/dinov2_vitl14_pretrain.pth',
    'vit_small_reg': 'vit/dinov2_vits14_reg4_pretrain.pth',
    'vit_large_reg': 'dinov2/dinov2_vitl14_reg4_pretrain.pth',
    'vit_giant2_reg': 'vit/dinov2_vitg14_reg4_pretrain.pth',
}