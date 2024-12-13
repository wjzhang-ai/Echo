import torch.nn as nn
import torch
import random
import torchvision.models as models_resnet
from timm.models.layers import trunc_normal_, DropPath
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from models.functions import *
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
 
)
from omegaconf import OmegaConf
from transformers import Wav2Vec2Model
import torch.nn.functional as F
from models.models_eco import UNetWrapper, EmbeddingAdapter
import os

from diffusers import AutoencoderKL
from models.ASPP_ASFF import*

class Upsample(nn.Module):  # this
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels,  # this conv let the size unchanged
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest"
        )  # double the size
        if self.with_conv:
            x = self.conv(x)
        return x


class EcoDepthEncoder(nn.Module):
    def __init__(
        self,
        out_dim=1024,
        ldm_prior=[32,64,256],
        sd_path=None,
        emb_dim=768,
        args=None,
    ):
        super().__init__()
 
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        self.apply(self._init_weights)

        self.cide_module = CIDE(emb_dim)

        self.config = OmegaConf.load("v1-inference.yaml")

       


    
        sd_model = instantiate_from_config(self.config.model)

    
        self.unet = UNetWrapper( sd_model.model, use_attn=False)
      
        self.aspp_asff=UNet_aspp_asff()
     


        del self.unet.unet.diffusion_model.out

      



 
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x = self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, audio_spec,audio_wave):
 
    
       
  
        latents=self.aspp_asff(audio_spec)

        
        conditioning_scene_embedding = self.cide_module(audio_wave)  
     

   
        t = torch.ones((audio_spec.shape[0],), device=audio_spec.device).long()


        outs = self.unet(latents, t, c_crossattn=[conditioning_scene_embedding])
      

        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]

        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        
        return self.out_layer(x)


class CIDE(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
 

       
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec.freeze_feature_extractor()  
        self.conv = nn.Conv1d(2, 1, kernel_size=1)
        self.fc = nn.Sequential(nn.Linear(768, 400), nn.GELU(), nn.Linear(400, 100))
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
  
        self.embeddings = nn.Parameter(torch.randn(100, self.dim))
   
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
  
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)




    def forward(self, x):

  

        x = self.conv(x) 
        x = x.squeeze(1)  
        
       
        mask_length = 10  
        min_input_length = self.wav2vec.config.inputs_to_logits_ratio * mask_length * 2
        if x.shape[1] < min_input_length:
            pad_length = min_input_length - x.shape[1]
            x = F.pad(x, (0, pad_length))

 
        with torch.no_grad():
            wav2vec_output = self.wav2vec(x).last_hidden_state  
  
 
 


        wav2vec_output = wav2vec_output.mean(dim=1)  
        class_probs = self.fc( wav2vec_output)
        class_probs = self.m(class_probs)


        class_embeddings = class_probs @ self.embeddings

        conditioning_scene_embedding = self.embedding_adapter(
            class_embeddings, self.gamma
        )

        return conditioning_scene_embedding


# 论文模型
class EcoDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_depth = 14.104
        # self.max_depth = 10


      
    
        embed_dim = 192
  
        channels_in = embed_dim * 8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset="nyu")
        self.decoder = Decoder(channels_in, channels_out)
       

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
        )

     

    def forward(self, audio_spec,audio_wave):
        #audio_wave[64,2,960]
        conv_feats = self.encoder(audio_spec,audio_wave)
      
        out = self.decoder([conv_feats])
        out_depth = self.last_layer_depth(out)

        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return out_depth
    


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = 3
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            3,
            [32,32,32],
            [2,2,2],
        )

        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        conv_layers.append(build_norm_layer(dict(type="BN"), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*conv_layers)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, conv_feats):
  
        out = self.deconv_layers(conv_feats[0])
        

        out = self.conv_layers(out)
       

    
        out = self.up(out)
        out = self.up(out)
  
        return out


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""

        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type="deconv"),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

 
    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

        return deconv_kernel, padding, output_padding



 
    