"""
Dataset handling for fingerprint embedding training.
"""
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FingerprintDataset(Dataset):
    """Dataset class for fingerprint images with augmentation support."""
    
    def __init__(self, root_dir, split='train', rotation_degrees=15, crop_percent=10):
        self.root_dir = os.path.join(root_dir, split)
        if split == 'train': 
            self.img_dir = os.path.join(self.root_dir, 'enhanced_merged_')
        else:
            self.img_dir = os.path.join(self.root_dir, 'enhanced')
        
        self.rotation_degrees = rotation_degrees
        self.crop_percent = crop_percent
        
        # Dynamic transform based on augmentation parameters
        transform_list = [transforms.Resize((256, 256))]
        
        # Only add augmentations if the parameters are > 0
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(
                degrees=rotation_degrees,
                fill=255  # White border completion
            ))
        
        if crop_percent > 0:
            # Add random crop followed by resize back to original size
            crop_size = int(256 * (1 - crop_percent/100))
            transform_list.append(transforms.RandomCrop(crop_size, padding=0, pad_if_needed=True, fill=255))
            transform_list.append(transforms.Resize((256, 256)))
        
        # Add final conversion to tensor
        transform_list.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transform_list)
        
        # Get all images and their IDs
        self.images = []
        self.ids = []
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith('.png'):
                self.images.append(img_name)
                
                # Extract ID based on filename format
                parts = img_name.split('_')
                
                # If format is like 0001_1_1.png, use R prefix
                if len(parts) > 2:
                    id_str = 'R' + parts[0]
                else:
                    id_str = parts[0]
                
                self.ids.append(id_str)
        
        # Create ID to index mapping
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(sorted(set(self.ids)))}
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        
        # Extract ID based on filename format
        parts = img_name.split('_')
        
        # If format is like 0001_1_1.png, use R prefix
        if len(parts) > 2:
            id_str = 'R' + parts[0]
        else:
            id_str = parts[0]
        
        label = self.id_to_idx[id_str]
        
        return image, label


def split_ids_for_validation(dataset, val_id_percent=20):
    """Split dataset IDs into validation and test sets."""
    # Get unique IDs
    unique_ids = list(set(dataset.ids))
    random.shuffle(unique_ids)
    
    # Calculate split point
    val_count = max(1, int(len(unique_ids) * val_id_percent / 100))
    
    # Split IDs
    val_ids = set(unique_ids[:val_count])
    test_ids = set(unique_ids[val_count:])
    
    # Create image indices for each set
    val_indices = [i for i, id_str in enumerate(dataset.ids) if id_str in val_ids]
    test_indices = [i for i, id_str in enumerate(dataset.ids) if id_str in test_ids]
    
    print(f"Split {len(unique_ids)} IDs into {len(val_ids)} validation IDs and {len(test_ids)} test IDs")
    print(f"This gives {len(val_indices)} validation samples and {len(test_indices)} test samples")
    
    return val_indices, test_indices


def create_data_loaders(config):
    """Create training and validation data loaders."""
    # Initialize datasets with augmentation
    train_dataset = FingerprintDataset(
        config.data_path, 
        'train',
        rotation_degrees=config.rotation_degrees,
        crop_percent=config.crop_percent
    )
    
    test_dataset = FingerprintDataset(
        config.data_path, 
        'test',
        rotation_degrees=0,  # No augmentation for test
        crop_percent=0
    )
    
    # Split test dataset into validation and test sets
    val_indices, test_indices = split_ids_for_validation(
        test_dataset, 
        val_id_percent=config.val_id_percent
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    return train_loader, test_dataset, val_indices, test_indices
