import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PositionEncoder(nn.Module):  # Nerf position encoder
    def __init__(self, output_dim, input_dim=2):
        super(PositionEncoder, self).__init__()
        self.input_dim = input_dim
        L_embed = int(output_dim / 2)
        self.L_embed = L_embed
        # Compute frequencies (2^i) for encoding
        self.freq_bands = 2.0 ** torch.linspace(0, L_embed - 1, L_embed)

    def forward(self, pos):
        # pos: [B, input_dim], e.g. [B,2]
        rets = []
        for freq in self.freq_bands:
            rets.append(torch.sin(freq * pos))
            rets.append(torch.cos(freq * pos))
        pos_enc = torch.cat(rets, dim=-1)  # [B, output_dim]
        return pos_enc


class NearbyEncoder(nn.Module):
    def __init__(self, pos_hidden_dim=12, num_heads=4, surround_panoramas=3):
        super(NearbyEncoder, self).__init__()

        self.surround_panoramas = surround_panoramas
        self.pos_hidden_dim = pos_hidden_dim

        base_resnet = models.resnet50(pretrained=True)
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_resnet.children())[1:-2], 
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.position_encoder = PositionEncoder(output_dim=pos_hidden_dim // 2, input_dim=2)

        self.panorama_encoder = nn.Sequential(
            *list(models.resnet50(pretrained=True).children())[:-2]
        )

        self.multi_attention = nn.MultiheadAttention(
            num_heads=num_heads, 
            embed_dim=2048 + pos_hidden_dim,
            kdim=2048 + pos_hidden_dim, 
            vdim=2048*16*16, 
            batch_first=True
        )

        self.attn_to_2d = nn.Linear(2048 + pos_hidden_dim, 2048 * 16 * 16)

        self.combined_hint_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )

    def forward(self, target_pos, target_structure, surround_pos, surround_structure, surround_panoramas, verbose=False):
        # target_pos: [batch_size, 2, 1, 1]
        if verbose: print("target_pos:", target_pos.shape)
        target_pos_cut = target_pos[:, :, 0, 0]   
        # [batch_size, 2]
        target_pos_features = self.position_encoder(target_pos_cut)  
        # [batch_size, pos_hidden_dim]
        if verbose: print("target_pos_features:", target_pos_features.shape)

        if verbose: print("target_structure:", target_structure.shape)
        target_structure_features = self.structure_encoder(target_structure)
        # [batch_size, 2048,1,1]
        target_structure_features = target_structure_features.squeeze(-1).squeeze(-1)
        # [batch_size,2048]
        if verbose: print("target_structure_features:", target_structure_features.shape)

        target_combined_features = torch.cat([target_structure_features, target_pos_features], dim=1)  
        # [batch_size, 2048 + pos_hidden_dim]
        target_combined_features = target_combined_features.unsqueeze(1)
        # [batch_size,1,2048+pos_hidden_dim]
        if verbose: print("target_combined_features:", target_combined_features.shape)

        # Surround
        # surround_structure: [batch_size, T, 1, 512,512]
        if verbose: print("surround_structure:", surround_structure.shape)
        batch_size, T, C, H, W = surround_structure.shape
        surround_structure_reshaped = surround_structure.view(batch_size * T, C, H, W)
        if verbose: print("surround_structure_reshaped:", surround_structure_reshaped.shape)
        surround_structure_features_reshaped = self.structure_encoder(surround_structure_reshaped)
        surround_structure_features_reshaped = surround_structure_features_reshaped.squeeze(-1).squeeze(-1)
        # [batch_size*T,2048]
        surround_structure_features = surround_structure_features_reshaped.view(batch_size, T, 2048)
        # [batch_size,T,2048]
        if verbose: print("surround_structure_features:", surround_structure_features.shape)

        # surround_pos: [batch_size,T,2,1,1]
        surround_pos_cut = surround_pos[:, :, 0, 0] # [batch_size,T * 2,2]
        if verbose: print("surround_pos_cut:", surround_pos_cut.shape)
        surround_pos_reshaped = surround_pos_cut.view(batch_size * T, 2)
        if verbose: print("surround_pos_reshaped:", surround_pos_reshaped.shape)
        surround_pos_features_reshaped = self.position_encoder(surround_pos_reshaped)  
        # [batch_size*T, pos_hidden_dim]
        surround_pos_features = surround_pos_features_reshaped.view(batch_size, T, self.pos_hidden_dim)
        # [batch_size,T,pos_hidden_dim]
        if verbose: print("surround_pos_features:", surround_pos_features.shape)

        surround_combined_features = torch.cat([surround_structure_features, surround_pos_features], dim=2)  
        # [batch_size,T,2048+pos_hidden_dim]
        if verbose: print("surround_combined_features:", surround_combined_features.shape)

        # Panoramas
        # surround_panoramas: [batch_size,T,3,512,512]
        panoramas_reshaped = surround_panoramas.view(batch_size * T, 3, 512, 512)
        panoramas_reshaped_features = self.panorama_encoder(panoramas_reshaped)
        # [batch_size*T,2048,16,16]
        panoramas_reshaped_features = panoramas_reshaped_features.view(batch_size, T, 2048*16*16)
        # V: [batch_size,T,2048*16*16]
        if verbose: print("panoramas_reshaped_features:", panoramas_reshaped_features.shape)

        # Multi-Head Attention
        # Q: [batch_size,1,2048+pos_hidden_dim]
        # K: [batch_size,T,2048+pos_hidden_dim]
        # V: [batch_size,T,2048*16*16]
        attn_output, attn_weights = self.multi_attention(
            query=target_combined_features,
            key=surround_combined_features,
            value=panoramas_reshaped_features
        )
        # attn_output: [batch_size,1,2048+pos_hidden_dim]
        if verbose: print("attn_output:", attn_output.shape)

        attn_output_1d = self.attn_to_2d(attn_output.squeeze(1))
        # [batch_size,2048*16*16]
        if verbose: print("attn_output_1d:", attn_output_1d.shape)

        attn_output_2d = attn_output_1d.view(batch_size, 2048,16,16)
        # [batch_size,2048,16,16]
        if verbose: print("attn_output_2d:", attn_output_2d.shape)

        decoded_output = self.combined_hint_decoder(attn_output_2d)
        # [batch_size,128,512,512]
        if verbose: print("decoded_output (Nearby):", decoded_output.shape)

        return decoded_output


class CrossViewEncoder(nn.Module):
    def __init__(self):
        super(CrossViewEncoder, self).__init__()
        base_resnet_sat = models.resnet50(pretrained=True)
        self.sat_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_resnet_sat.children())[1:-2]
        )

        base_resnet_str = models.resnet50(pretrained=True)
        self.str_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_resnet_str.children())[1:-2]
        )
        # 输出 [batch_size,2048,16,16]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 2048),
            nn.ReLU(),

            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),

            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )

    def forward(self, target_structure, target_satellite, verbose=False):
        # target_satellite: [batch_size,3,512,512]
        # target_structure: [batch_size,1,512,512]

        sat_feat = self.sat_encoder(target_satellite)      
        # [batch_size,2048,16,16]
        str_feat = self.str_encoder(target_structure)      
        # [batch_size,2048,16,16]

        if verbose: 
            print("sat_feat:", sat_feat.shape)
            print("str_feat:", str_feat.shape)
    
        combined_feat = torch.cat([sat_feat, str_feat], dim=1)
        # [batch_size,4096,16,16]
        if verbose: print("combined_feat (CrossView):", combined_feat.shape)

        out = self.decoder(combined_feat)
        # [batch_size,128,512,512]
        if verbose: print("crossview_decoder_output:", out.shape)

        return out


class MixedViewDiff(nn.Module):
    def __init__(self):
        super(MixedViewDiff, self).__init__()
        self.nearby_encoder = NearbyEncoder()
        self.crossview_encoder = CrossViewEncoder()
        self.view_result = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, target_pos, target_structure, target_satellite, surround_pos, surround_structure, surround_panoramas, verbose=False, view_result=False):
        crossview_embedding = self.crossview_encoder(target_structure, target_satellite, verbose=verbose)  
        # [batch_size,128,512,512]
        nearby_embedding = self.nearby_encoder(target_pos, target_structure, surround_pos, surround_structure, surround_panoramas, verbose=verbose)
        # [batch_size,128,512,512]

        if verbose: 
            print("crossview_embedding:", crossview_embedding.shape)
            print("nearby_embedding:", nearby_embedding.shape)

        output = torch.cat([crossview_embedding, nearby_embedding], dim=1)

        if view_result:
            output = self.view_result(output)
        # [batch_size,256,512,512]
        if verbose: print("final_output:", output.shape)
        return output


def main(verbose=True):
    batch_size = 2
    T = 3
    target_pos = torch.randn(batch_size, 2, 512, 512)            # [batch_size,2,512,512]
    target_structure = torch.randn(batch_size, 1, 512, 512)  # [batch_size,1,512,512]
    target_satellite = torch.randn(batch_size, 3, 512,512)    # [batch_size,3,512,512]
    surround_pos = torch.randn(batch_size, T * 2,1,1)          # [batch_size,T * 2,1,1]
    surround_structure = torch.randn(batch_size, T, 1,512,512)# [batch_size,T,1,512,512]
    surround_panoramas = torch.randn(batch_size, T, 3,512,512)# [batch_size,T,3,512,512]

    model = MixedViewDiff()
    output = model(target_pos, target_structure, target_satellite, surround_pos, surround_structure, surround_panoramas, verbose=verbose)
    if verbose: print("output:", output.shape)

if __name__ == "__main__":
    main(verbose=True)