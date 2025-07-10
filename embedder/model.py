"""
Embedder model and loss function for fingerprint embedding training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class EmbeddingNet(nn.Module):
    """Neural network for fingerprint embedding extraction."""
    
    def __init__(self, embedding_dim=512, depth=9, filters_base=16):
        super(EmbeddingNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.depth = depth
        self.filters_base = filters_base

        # Build dynamic architecture based on parameters
        layers = []
        in_channels = 1  # Grayscale input
        current_filters = self.filters_base

        # Create convolutional blocks based on depth
        for i in range(self.depth):
            # Double the number of filters after each block (until a maximum)
            out_channels = min(current_filters * (2**i), 512)

            # Add conv block
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))

            # Every 2 blocks, add pooling to reduce spatial dimensions
            if (i + 1) % 2 == 0 or i == 0:
                layers.append(nn.MaxPool2d(2))

            in_channels = out_channels

        # Final layers
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*layers)

        # Fully connected layer for embedding
        self.fc = nn.Linear(in_channels, self.embedding_dim)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, embed_config=None, device=None):
        """Load a model from checkpoint using the provided config."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get the state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create model based on embed_config or defaults
        if embed_config:
            model = cls(
                embedding_dim=embed_config.get('embedding_dim', 512),
                depth=embed_config.get('depth', 9),
                filters_base=embed_config.get('filters_base', 16)
            ).to(device)
        else:
            # Use defaults
            model = cls().to(device)

        # Load state dict
        model.load_state_dict(state_dict)
        return model


class TripletLoss(nn.Module):
    """Triplet loss for embedding learning."""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # Calculate pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2).pow(2)
        
        # Create positive and negative masks
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        
        # Don't compare elements with themselves
        pos_mask.fill_diagonal_(False)
        
        # Add large value to diagonal to exclude self-comparisons
        diagonal_mask = torch.eye(len(embeddings), device=embeddings.device) * 1e9
        pairwise_dist = pairwise_dist + diagonal_mask
        
        # Find hardest positive and negative examples
        hardest_positive = (pairwise_dist * pos_mask.float()).max(dim=1)[0]
        hardest_negative = (pairwise_dist * neg_mask.float()).min(dim=1)[0]
        
        # Calculate triplet loss
        loss = F.relu(hardest_positive - hardest_negative + self.margin)
        return loss.mean()
