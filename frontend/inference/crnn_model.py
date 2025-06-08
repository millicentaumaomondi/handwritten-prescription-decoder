import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models  # This is the missing import

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x

# --- CBAM Modules ---
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        out = torch.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.sigmoid(self.conv(pooled))
        return x * attention_map

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# --- CRNN with efficientnet_b4 Backbone ---
# --- Enhanced Model Definition with Regularization ---
class CRNN_EfficientNet(nn.Module):
    def __init__(self, input_channels=1, hidden_size=256, num_classes=43, num_lstm_layers=3, dropout_rate=0.3):
        super(CRNN_EfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B4
        efficientnet = models.efficientnet_b4(pretrained=True)
        
        # Modify first layer for grayscale (if input_channels=1)
        if input_channels == 1:
            efficientnet.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Use all features except the classifier
        self.feature_extractor = nn.Sequential(
            *list(efficientnet.features.children()),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Dropout2d(dropout_rate/2)  # Spatial dropout
        )
        
        self.cbam = CBAM(1792)  # EfficientNet-B4 final channels: 1792
        
        # Infer LSTM input size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 64, 256)
            feat = self.feature_extractor(dummy)
            _, c, h, w = feat.shape
            self.lstm_input_size = c * h
        
        self.pos_enc = PositionalEncoding(self.lstm_input_size)
        
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            hidden_size,
            num_lstm_layers,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        # Additional dropout before final classification
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.cbam(features)
        
        # Reshape for LSTM (Seq-first: [T, B, F])
        b, c, h, w = features.shape
        features = features.permute(0, 3, 1, 2).reshape(b, w, -1)
        features = features.permute(1, 0, 2)  # [T, B, F]
        features = self.pos_enc(features)
        
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits

