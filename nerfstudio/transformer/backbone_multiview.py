import torch
import torch.nn as nn
from einops import rearrange

from nerfstudio.transformer.backbone_cnn import CNNEncoder
from nerfstudio.transformer.multiview_tranformer import MultiViewFeatureTransformer
from nerfstudio.transformer.utils import split_feature, merge_splits
from nerfstudio.transformer.position import PositionEmbeddingSine


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
        no_split_still_shift=False,
        no_cross_attn=False,
        global_attn_fast=True,
        downscale_factor=8,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )  

        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        

       
    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)   ## CNN encoder, 4× downsampling
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features

        cur_features_list = [x[0] for x in features_list]

        # add position to features
        cur_features_list = feature_add_position_list(cur_features_list, attn_splits, self.feature_channels)

        # Transformer
        cur_features_list = self.transformer(
            cur_features_list, attn_num_splits=attn_splits)

        features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]
        features = rearrange(features, "b v ... -> (v b) ...")
        ## 4× upsample to ori resolution
        trans_feature = self.upsampler(features)
        return trans_feature

if __name__ == "__main__":
    backbone = BackboneMultiview(feature_channels=128,downscale_factor=4,no_cross_attn=True).cuda()
    input_tensor = torch.rand(1,2,3,176*2,704*2).cuda() #[B,N,C,H,W]
    out = backbone(input_tensor,attn_splits=2)
    print(f"output shape: {out.shape}")