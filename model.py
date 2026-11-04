import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

class DenseBlock(nn.Module):
    """Dense block for feature extraction"""
    
    def __init__(self, in_channels, growth_rate=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_channels = in_channels
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(curr_channels, growth_rate, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            curr_channels += growth_rate
            
        # Transition layer to reduce channels back to input size
        self.transition = nn.Sequential(
            nn.Conv2d(curr_channels, in_channels, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return self.transition(torch.cat(features, 1))

class DFA(nn.Module):
    """Autoencoder for image enhancement"""
    
    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Initialize VGG for perceptual loss
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        encoder_layers = []
        in_channels = 3
        channels = 32
        encoder_channels = []
        decoder_channels = []
        
        # Calculate channels
        curr_channels = channels
        for i in range(num_layers):
            encoder_channels.append(curr_channels)
            curr_channels = min(curr_channels * 2, 512)
            
        curr_channels = encoder_channels[-1]
        for i in range(num_layers-1):
            out_channels = max(curr_channels // 2, 32)
            decoder_channels.append(out_channels)
            curr_channels = out_channels
        
        # Create encoder with dense blocks
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    DenseBlock(channels)
                ))
            else:
                encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, 2, 1),
                    DenseBlock(channels)
                ))
            in_channels = channels
            channels = min(channels * 2, 512)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Create decoder with residual connections
        decoder_layers = []
        channels = encoder_channels[-1]
        for i in range(num_layers):
            if i == num_layers - 1:
                decoder_layers.append(nn.Sequential(
                    nn.Conv2d(channels + 3, 64, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 3, 3, 1, 1),
                    nn.Sigmoid()
                ))
            else:
                out_channels = max(channels // 2, 32)
                decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(channels, out_channels, 4, 2, 1),
                    DenseBlock(out_channels)
                ))
                channels = out_channels
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Projection layers for tap loss
        self.enc_proj_layers = nn.ModuleList()
        self.dec_proj_layers = nn.ModuleList()
        
        for i in range(min(3, num_layers-1)):
            enc_channels = encoder_channels[i]
            dec_channels = decoder_channels[-(i+1)]
            
            self.enc_proj_layers.append(nn.Sequential(
                nn.Conv2d(enc_channels, enc_channels//2, 1),
                nn.ReLU(True),
                nn.Conv2d(enc_channels//2, 3, 1)
            ))
            
            self.dec_proj_layers.append(nn.Sequential(
                nn.Conv2d(dec_channels, dec_channels//2, 1),
                nn.ReLU(True),
                nn.Conv2d(dec_channels//2, 3, 1)
            ))
    
    def get_vgg_features(self, x):
        """Extract VGG features for perceptual loss"""
        return self.vgg_features(x)
    
    def encode(self, x):
        """Encode input image"""
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return x, features
    
    def decode(self, x, input_img=None):
        """Decode encoded features"""
        features = []
        for i, layer in enumerate(self.decoder_layers):
            if i == len(self.decoder_layers) - 1 and input_img is not None:
                x = torch.cat([x, input_img], dim=1)
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                features.append(x)
        return x, features
    
    def forward(self, degraded, clean=None):
        """Forward pass"""
        # Encode degraded input
        x_deg, enc_deg_feats = self.encode(degraded)
        
        # Encode clean input if provided (for training)
        enc_clean_feats = []
        if clean is not None:
            _, enc_clean_feats = self.encode(clean)
        
        # Decode
        out, dec_feats = self.decode(x_deg, degraded)
        
        if clean is None:
            return out, None  # Inference mode
        else:
            return out, None, enc_clean_feats, dec_feats  # Training mode

class LAE(nn.Module):
    """Autoencoder for image enhancement"""
    
    def __init__(self, num_layers=4, latent_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Initialize VGG for perceptual loss
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        encoder_layers = []
        in_channels = 3
        channels = 32
        encoder_channels = []
        decoder_channels = []
        
        # Calculate channels
        curr_channels = channels
        for i in range(num_layers):
            encoder_channels.append(curr_channels)
            curr_channels = min(curr_channels * 2, 512)
            
        curr_channels = encoder_channels[-1]
        for i in range(num_layers-1):
            out_channels = max(curr_channels // 2, 32)
            decoder_channels.append(out_channels)
            curr_channels = out_channels
        
        # Create encoder with dense blocks
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    DenseBlock(channels)
                ))
            else:
                encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, 2, 1),
                    DenseBlock(channels)
                ))
            in_channels = channels
            channels = min(channels * 2, 512)
        self.encoder_layers = nn.ModuleList(encoder_layers)

        # Create latent head
        self.latent_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(encoder_channels[-1], latent_dim)  # (B, D), D > 2
        )
        self.latent_to_feature = nn.Sequential(
            nn.Linear(latent_dim, encoder_channels[-1]),
            nn.ReLU(True)
        )
        
        # Create decoder with residual connections
        decoder_layers = []
        channels = encoder_channels[-1]
        for i in range(num_layers):
            if i == num_layers - 1:
                decoder_layers.append(nn.Sequential(
                    nn.Conv2d(channels + 3, 64, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 3, 3, 1, 1),
                    nn.Sigmoid()
                ))
            else:
                out_channels = max(channels // 2, 32)
                decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(channels, out_channels, 4, 2, 1),
                    DenseBlock(out_channels)
                ))
                channels = out_channels
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Projection layers for tap loss
        self.enc_proj_layers = nn.ModuleList()
        self.dec_proj_layers = nn.ModuleList()
        
        for i in range(min(3, num_layers-1)):
            enc_channels = encoder_channels[i]
            dec_channels = decoder_channels[-(i+1)]
            
            self.enc_proj_layers.append(nn.Sequential(
                nn.Conv2d(enc_channels, enc_channels//2, 1),
                nn.ReLU(True),
                nn.Conv2d(enc_channels//2, 3, 1)
            ))
            
            self.dec_proj_layers.append(nn.Sequential(
                nn.Conv2d(dec_channels, dec_channels//2, 1),
                nn.ReLU(True),
                nn.Conv2d(dec_channels//2, 3, 1)
            ))
    
    def get_vgg_features(self, x):
        """Extract VGG features for perceptual loss"""
        return self.vgg_features(x)
    
    def encode(self, x):
        """Encode input image"""
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return x, features
    
    def decode(self, x, input_img=None):
        """Decode encoded features"""
        features = []
        for i, layer in enumerate(self.decoder_layers):
            if i == len(self.decoder_layers) - 1 and input_img is not None:
                x = torch.cat([x, input_img], dim=1)
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                features.append(x)
        return x, features
    
    def forward(self, degraded, clean=None):
        """Forward pass"""
        x_deg, _ = self.encode(degraded)
        z_deg = self.latent_head(x_deg)

        B, C, H, W = x_deg.shape
        x_latent = self.latent_to_feature(z_deg)
        x_latent = x_latent.view(B, C, 1, 1)
        x_latent = x_latent.expand(-1, -1, H, W)
        
        enc_clean_feats = []
        if clean is not None:
            _, enc_clean_feats = self.encode(clean)
        
        # Decode
        out, dec_feats = self.decode(x_latent, degraded)
        
        if clean is None:
            return out, z_deg  # Inference mode
        else:
            return out, z_deg, enc_clean_feats, dec_feats  # Training mode

class DLAE(nn.Module):
    """Autoencoder for image enhancement"""
    
    def __init__(self, num_layers=4, num_classes=10, latent_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Initialize VGG for perceptual loss
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        encoder_layers = []
        in_channels = 3
        channels = 32
        encoder_channels = []
        decoder_channels = []
        
        # Calculate channels
        curr_channels = channels
        for i in range(num_layers):
            encoder_channels.append(curr_channels)
            curr_channels = min(curr_channels * 2, 512)
            
        curr_channels = encoder_channels[-1]
        for i in range(num_layers-1):
            out_channels = max(curr_channels // 2, 32)
            decoder_channels.append(out_channels)
            curr_channels = out_channels
        
        # Create encoder with dense blocks
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    DenseBlock(channels)
                ))
            else:
                encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, 2, 1),
                    DenseBlock(channels)
                ))
            in_channels = channels
            channels = min(channels * 2, 512)
        self.encoder_layers = nn.ModuleList(encoder_layers)

        bottleneck_channels = encoder_channels[-1]

        self.latent_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, latent_dim)
        )

        self.latent_to_feature = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_channels),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Create decoder with residual connections
        decoder_layers = []
        channels = encoder_channels[-1]
        for i in range(num_layers):
            if i == num_layers - 1:
                decoder_layers.append(nn.Sequential(
                    nn.Conv2d(channels + 3, 64, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 3, 3, 1, 1),
                    nn.Sigmoid()
                ))
            else:
                out_channels = max(channels // 2, 32)
                decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(channels, out_channels, 4, 2, 1),
                    DenseBlock(out_channels)
                ))
                channels = out_channels
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Projection layers for tap loss
        self.enc_proj_layers = nn.ModuleList()
        self.dec_proj_layers = nn.ModuleList()
        
        for i in range(min(3, num_layers-1)):
            enc_channels = encoder_channels[i]
            dec_channels = decoder_channels[-(i+1)]
            
            self.enc_proj_layers.append(nn.Sequential(
                nn.Conv2d(enc_channels, enc_channels//2, 1),
                nn.ReLU(True),
                nn.Conv2d(enc_channels//2, 3, 1)
            ))
            
            self.dec_proj_layers.append(nn.Sequential(
                nn.Conv2d(dec_channels, dec_channels//2, 1),
                nn.ReLU(True),
                nn.Conv2d(dec_channels//2, 3, 1)
            ))
    
    def get_vgg_features(self, x):
        """Extract VGG features for perceptual loss"""
        return self.vgg_features(x)
    
    def encode(self, x):
        """Encode input image"""
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return x, features
    
    def decode(self, x, input_img=None):
        """Decode encoded features"""
        features = []
        for i, layer in enumerate(self.decoder_layers):
            if i == len(self.decoder_layers) - 1 and input_img is not None:
                x = torch.cat([x, input_img], dim=1)
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                features.append(x)
        return x, features
    
    def forward(self, degraded, clean=None):
        """Forward pass"""
        # Encode degraded input
        x_deg, _ = self.encode(degraded)

        # Encode clean input if provided (for training)
        enc_clean_feats = []
        if clean is not None:
            _, enc_clean_feats = self.encode(clean)

        z = self.latent_head(x_deg)

        B, _, H, W = x_deg.shape
        x_latent = self.latent_to_feature(z)
        C_latent = x_latent.shape[1]
        x_latent = x_latent.view(B, C_latent, 1, 1)
        x_latent = x_latent.expand(-1, -1, H, W)

        x = x_deg + x_latent

        logits = self.classifier(z)

        # Decode
        out, dec_feats = self.decode(x, degraded) # the old model is in the old_training file
        
        if clean is None:
            return out, z, logits  # Inference mode
        else:
            return out, z, logits, enc_clean_feats, dec_feats  # Training mode
