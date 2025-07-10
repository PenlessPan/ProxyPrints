"""
Training logic for fingerprint embedder.
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from .model import EmbeddingNet, TripletLoss
from .dataset import create_data_loaders
from .utils import EarlyStopping

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc="Training") as progress_bar:
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(images)
            
            # Calculate loss
            loss = criterion(embeddings, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_embeddings(model, val_dataset, val_indices, device):
    """Simple validation using embedding distances."""
    model.eval()
    
    # Extract embeddings for validation samples
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for idx in val_indices:
            image, label = val_dataset[idx]
            image = image.unsqueeze(0).to(device)
            embedding = model(image)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Calculate intra-class vs inter-class distances
    intra_distances = []
    inter_distances = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if labels[i] == labels[j]:
                intra_distances.append(dist)
            else:
                inter_distances.append(dist)
    
    # Calculate metrics
    mean_intra = np.mean(intra_distances) if intra_distances else 0
    mean_inter = np.mean(inter_distances) if inter_distances else 1
    
    # Separation ratio (higher is better)
    separation_ratio = mean_inter / mean_intra if mean_intra > 0 else 0
    
    return {
        'mean_intra_distance': mean_intra,
        'mean_inter_distance': mean_inter,
        'separation_ratio': separation_ratio
    }


def visualize_embeddings(model, dataset, indices, device, epoch, config):
    """Create t-SNE visualization of embeddings."""
    model.eval()
    
    # Limit to 500 samples for visualization
    indices_subset = indices[:min(len(indices), 500)]
    
    all_embeddings = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for idx in indices_subset:
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            embedding = model(image).cpu().numpy()
            all_embeddings.append(embedding[0])
            all_labels.append(label)
            all_ids.append(dataset.ids[idx])
            
    all_embeddings = np.vstack(all_embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot embeddings colored by ID
    plt.figure(figsize=(10, 10))
    
    # Convert IDs to numeric values for coloring
    unique_ids = list(set(all_ids))
    id_to_num = {id_: i for i, id_ in enumerate(unique_ids)}
    numeric_ids = [id_to_num[id_] for id_ in all_ids]
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=numeric_ids, cmap='tab20', alpha=0.6, s=10)
    plt.title(f'Embedding t-SNE - Epoch {epoch}')
    
    # Save the plot
    plt.tight_layout()
    tsne_path = os.path.join(config.log_dir, f'tsne_epoch_{epoch}.png')
    plt.savefig(tsne_path)
    plt.close()
    
    return tsne_path


def train_model(config, use_wandb=False):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = EmbeddingNet(
        embedding_dim=config.embedding_dim,
        depth=config.conv_layers_depth,
        filters_base=config.filters_base
    ).to(device)
    
    # Initialize loss function and optimizer
    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create data loaders
    train_loader, test_dataset, val_indices, test_indices = create_data_loaders(config)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_epochs=config.min_epochs_before_stopping
    )
    
    # Training loop
    best_separation_ratio = 0
    
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_embeddings(model, test_dataset, val_indices, device)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, "
                   f"Separation Ratio: {val_metrics['separation_ratio']:.4f}")
        
        # Optional: log to wandb
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "separation_ratio": val_metrics['separation_ratio'],
                "mean_intra_distance": val_metrics['mean_intra_distance'],
                "mean_inter_distance": val_metrics['mean_inter_distance']
            })
        
        # Create t-SNE visualization every 5 epochs
        if epoch % 5 == 0:
            try:
                visualization_path = visualize_embeddings(model, test_dataset, val_indices, device, epoch, config)
                logger.info(f"Created t-SNE visualization: {visualization_path}")
            except Exception as e:
                logger.warning(f"Failed to create visualization: {e}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch}.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'config': vars(config),
            'model_config': {
                'embedding_dim': model.embedding_dim,
                'depth': model.depth,
                'filters_base': model.filters_base
            }
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Check for early stopping
        should_stop, is_best = early_stopping(val_metrics['separation_ratio'], epoch)
        
        if is_best:
            best_separation_ratio = val_metrics['separation_ratio']
            # Save best model separately
            best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved new best model with separation ratio: {best_separation_ratio:.4f}")
        
        if should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("Training completed.")
    logger.info(f"Best separation ratio: {best_separation_ratio:.4f}")
    
    # Load best model for return
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_separation_ratio
