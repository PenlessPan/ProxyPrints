import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
import hashlib
from pathlib import Path

# Add StyleGAN path to system path - use relative path to repo structure
def get_stylegan_path():
    """Get the StyleGAN path relative to the current file location"""
    current_dir = Path(__file__).parent
    stylegan_path = current_dir / "StyleGAN.pytorch"
    return str(stylegan_path)

# Add StyleGAN path to system path
stylegan_path = get_stylegan_path()
sys.path.insert(0, stylegan_path)

try:
    # Import StyleGAN generator
    from models.GAN import Generator
    from config import cfg as opt
except ImportError as e:
    raise ImportError(f"Failed to import StyleGAN modules. Make sure StyleGAN.pytorch is in the correct path: {stylegan_path}") from e

class EmbeddingNet(nn.Module):
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

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, embed_config=None, device=None):
        """
        Load a model from checkpoint using the provided config
        """
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

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """Adjust the dynamic range of tensor data"""
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)

class ProxyPrints:
    # Model download URLs and information
    MODEL_INFO = {
        "embedder.pth": "Trained embedder model for fingerprint feature extraction",
        "generator.pth": "Trained StyleGAN generator for fingerprint synthesis",
        "config": "fingerprints_config.yaml - StyleGAN configuration for fingerprint generation"
    }
    
    def __init__(self, embedder_name="embedder.pth", generator_name="generator.pth", 
                 config_name="fingerprints_config.yaml", embed_config=None, seed=24, key="default_key"):
        """
        Initialize ProxyPrints system.
        
        Args:
            embedder_name: Name of embedder model file in models/ directory
            generator_name: Name of StyleGAN generator file in models/ directory  
            config_name: Name of config file in StyleGAN.pytorch/configs/ directory
            embed_config: Configuration for embedder architecture
            seed: Random seed for reproducibility
            key: Key for rotation transformation
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.rotation_key = key  # Store rotation key for alignment
        
        # Setup model paths
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize embedder
        embedder_path = self.models_dir / embedder_name
        if embedder_path.exists():
            print(f"Loading embedder from {embedder_path}")
            self.embedder = EmbeddingNet.load_from_checkpoint(
                str(embedder_path), 
                embed_config=embed_config,
                device=self.device
            )
        elif embed_config is not None:
            print(f"Creating new embedder with config: {embed_config}")
            self.embedder = EmbeddingNet(
                embedding_dim=embed_config.get('embedding_dim', 512),
                depth=embed_config.get('depth', 9),
                filters_base=embed_config.get('filters_base', 16)
            ).to(self.device)
        else:
            print("Warning: No embedder found and no config provided. Using default architecture.")
            print(f"Place your embedder model at: {embedder_path}")
            self.embedder = EmbeddingNet().to(self.device)

        self.embedder.eval()

        # Initialize StyleGAN generator
        generator_path = self.models_dir / generator_name
        self._initialize_generator(generator_path, config_name)

        # Setup image transforms
        self.transform_image = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        if self.generator is not None:
            self.initialize_fixed_noise()

    def _initialize_generator(self, generator_path, config_name):
        """Initialize the StyleGAN generator with proper configuration"""
        try:
            # Build config path
            config_path = os.path.join(stylegan_path, "configs", config_name)
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}\nExpected: fingerprints_config.yaml")
            
            opt.merge_from_file(config_path)
            opt.freeze()

            self.generator = Generator(
                resolution=opt.dataset.resolution,
                num_channels=opt.dataset.channels,
                structure=opt.structure,
                **opt.model.gen
            ).to(self.device)

            if generator_path.exists():
                self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
                print(f"Loaded generator from {generator_path}")
            else:
                print(f"Warning: Generator not found at {generator_path}")
                print("Please download the generator model and place it in the models/ directory")
                print("You can download models from the releases or train your own")
            
            self.generator.eval()
            self.out_depth = int(np.log2(opt.dataset.resolution)) - 2
            self.latent_size = opt.model.gen.latent_size
            
        except Exception as e:
            print(f"Error initializing generator: {e}")
            self.generator = None
            self.out_depth = None
            self.latent_size = None

    def initialize_fixed_noise(self):
        """Generate and set fixed noise for all NoiseLayer instances in the generator"""
        if self.generator is None:
            return
            
        from models.CustomLayers import NoiseLayer

        # Set seed for reproducibility
        current_torch_state = torch.get_rng_state()

        try:
            torch.manual_seed(self.seed)
            
            # Find all NoiseLayer instances in the generator
            def find_noise_layers(module, prefix=''):
                noise_layers = []
                for name, child in module.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name
                    if isinstance(child, NoiseLayer):
                        noise_layers.append((full_name, child))
                    else:
                        noise_layers.extend(find_noise_layers(child, full_name))
                return noise_layers
            
            noise_layers = find_noise_layers(self.generator.g_synthesis)
            
            # Generate and set fixed noise for each layer
            for name, layer in noise_layers:
                # Generate a test input to determine the right noise shape
                with torch.no_grad():
                    test_input = torch.ones(1, self.generator.g_mapping.dlatent_size).to(self.device)
                    dummy_latent = self.generator.g_mapping(test_input)
                    
                    # Forward pass with hooks to capture noise shapes
                    shapes = {}
                    hooks = []
                    
                    def hook_fn(m, input, output):
                        if isinstance(m, NoiseLayer):
                            # Get the shape from the input tensor
                            shapes[m] = (input[0].shape[0], 1, input[0].shape[2], input[0].shape[3])
                    
                    for _, l in noise_layers:
                        hooks.append(l.register_forward_hook(hook_fn))
                    
                    # Do a forward pass to capture shapes
                    _ = self.generator.g_synthesis(dummy_latent, depth=self.out_depth, alpha=1)
                    
                    # Remove hooks
                    for h in hooks:
                        h.remove()
                    
                    # Set fixed noise for each layer
                    for _, l in noise_layers:
                        if l in shapes:
                            l.noise = torch.randn(shapes[l], device=self.device)
                            
        except Exception as e:
            print(f"Warning: Could not initialize fixed noise: {e}")
        finally:
            torch.set_rng_state(current_torch_state)

    def embed(self, input_image):
        """
        Get the embedding from an input fingerprint image.
        
        Args:
            input_image: Can be a path (str), PIL Image, or torch.Tensor
            
        Returns:
            torch.Tensor: Normalized embedding vector
        """
        self.embedder.eval()

        # Process input image
        if isinstance(input_image, str):
            # Load image from path
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Input image path does not exist: {input_image}")
            image = Image.open(input_image).convert('L')
            image = self.transform_image(image).unsqueeze(0).to(self.device)
        elif isinstance(input_image, Image.Image):
            # Process PIL image
            image = input_image.convert('L')
            image = self.transform_image(image).unsqueeze(0).to(self.device)
        elif isinstance(input_image, torch.Tensor):
            # Assume it's already a tensor
            image = input_image.to(self.device)
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
        else:
            raise TypeError("Input must be a path, PIL Image, or torch.Tensor")

        with torch.no_grad():
            embedding = self.embedder(image)

        return embedding

    def _generate_rotation_angles(self, key, embedding_dim):
        """
        Generate consistent rotation angles based on a key.
        
        Args:
            key: String key for generating consistent rotations
            embedding_dim: Dimension of the embedding space
            
        Returns:
            numpy.ndarray: Array of rotation angles
        """
        # Hash the key to get consistent byte sequence
        hash_object = hashlib.sha256(key.encode())
        hash_bytes = hash_object.digest()

        # Convert hash bytes to a seed for random number generation
        seed = int.from_bytes(hash_bytes[:4], byteorder='big')

        # Create a separate random number generator
        rng = np.random.RandomState(seed)

        # Determine how many rotation pairs we need (pairs of dimensions to rotate)
        num_rotation_pairs = embedding_dim - 1 

        # Generate rotation angles in radians (0 to 2π)
        rotation_angles = rng.uniform(0, 2 * np.pi, num_rotation_pairs)

        return rotation_angles

    def align(self, embedding, key=None):
        """
        Apply rotation transformation to the embedding based on the key.
        
        Args:
            embedding: Input embedding tensor
            key: Key for rotation (uses instance key if None)
            
        Returns:
            torch.Tensor: Rotated embedding
        """
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("Embedding must be a torch.Tensor")

        if key is None:
            key = self.rotation_key

        # Get embedding dimensions
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension if missing

        batch_size, embedding_dim = embedding.shape

        # Generate rotation angles
        rotation_angles = self._generate_rotation_angles(key, embedding_dim)

        # Create a copy of the embedding to transform
        transformed_embedding = embedding.clone()

        # Apply rotations to overlapping pairs (0,1), (1,2), (2,3), etc.
        for i, angle in enumerate(rotation_angles):
            # Get indices for the pair of dimensions to rotate
            dim1 = i
            dim2 = i + 1

            # Extract the coordinates for this pair
            x = transformed_embedding[:, dim1].clone()
            y = transformed_embedding[:, dim2].clone()

            # Apply rotation
            transformed_embedding[:, dim1] = x * np.cos(angle) - y * np.sin(angle)
            transformed_embedding[:, dim2] = x * np.sin(angle) + y * np.cos(angle)

        return transformed_embedding

    def generate(self, embedding):
        """
        Generate a fingerprint image from a single embedding vector.
        
        Args:
            embedding: Input embedding tensor
            
        Returns:
            PIL.Image: Generated fingerprint image
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Please provide a valid generator_path and config_path.")

        with torch.no_grad():
            # Ensure embedding is on the correct device
            embedding = embedding.to(self.device)
                        
            # Ensure proper dimensions - make it a batch of size 1 if it's just a vector
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            
            # Generate image using StyleGAN
            generated_image = self.generator(embedding, depth=self.out_depth, alpha=1)
            
            # Adjust dynamic range
            generated_image = adjust_dynamic_range(generated_image)
            
            # Convert to PIL image
            generated_image = generated_image.cpu().squeeze()
            generated_image = transforms.ToPILImage()(generated_image)
        
    def transform(self, input_image, key=None):
        """
        Process through the full pipeline: embedder -> aligner -> generator
        
        Args:
            input_image: Path to input fingerprint image, PIL Image, or torch.Tensor
            key: Key for rotation (uses instance key if None)
            
        Returns:
            PIL.Image: Transformed fingerprint image
        """
        # Step 1: Embed - get the embedding from the input image
        embedding = self.embed(input_image)

        # Step 2: Align - apply rotation transformation based on the key
        aligned_embedding = self.align(embedding, key)

        # Step 3: Generate - create a new fingerprint from the aligned embedding
        generated_image = self.generate(aligned_embedding)

        return generated_image

    def trans_nr(self, input_image):
        """
        Process through the pipeline without rotation: embedder -> generator
        
        Args:
            input_image: Path to input fingerprint image, PIL Image, or torch.Tensor
            
        Returns:
            PIL.Image: Transformed fingerprint image (no rotation)
        """
        # Step 1: Embed - get the embedding from the input image
        embedding = self.embed(input_image)

        # Step 2: Generate - create a new fingerprint from the embedding
        generated_image = self.generate(embedding)

        return generated_image

    @staticmethod
    def print_model_info():
        """Print information about required models and how to obtain them"""
        print("ProxyPrints Model Requirements:")
        print("=" * 50)
        print("\nRequired files in models/ directory:")
        for filename, description in ProxyPrints.MODEL_INFO.items():
            if filename != "config":
                print(f"  {filename}: {description}")
        
        print(f"\nRequired config file: {ProxyPrints.MODEL_INFO['config']}")
        
        print("\nTo obtain models:")
        print("1. Download from GitHub releases")
        print("2. Train your own using the provided training scripts")
        print("3. Place model files in the models/ directory")
        print("\nExample usage:")
        print("  proxy = ProxyPrints()")
        print("  proxy = ProxyPrints(embedder_name='my_embedder.pth', generator_name='my_generator.pth')")
        
    @staticmethod
    def list_available_models():
        """List models available in the models directory"""
        models_dir = Path(__file__).parent / "models"
        if not models_dir.exists():
            print("Models directory does not exist. It will be created on first use.")
            return []
        
        model_files = [f for f in models_dir.iterdir() if f.is_file() and f.suffix in ['.pth', '.pt']]
        if model_files:
            print("Available models:")
            for model in model_files:
                print(f"  {model.name}")
        else:
            print("No models found in models/ directory")
            print("Use ProxyPrints.print_model_info() for setup instructions")
        
        return model_files
        
no rotation)
        """
        # Step 1: Embed - get the embedding from the input image
        embedding = self.embed(input_image_path)

        # Step 2: Generate - create a new fingerprint from the embedding
        generated_image = self.generate(embedding)

        return generated_image