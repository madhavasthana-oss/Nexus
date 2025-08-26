import os
import torch
import PIL
from PIL import Image

import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class readRGBImages(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = ('.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.tif')
        
        # Validate root directory
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
        
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"Path is not a directory: {root_dir}")
        
        # Get list of valid image files
        try:
            all_files = os.listdir(self.root_dir)
            self.img_list = [
                img for img in all_files 
                if img.lower().endswith(self.valid_extensions)
            ]
        except PermissionError as e:
            raise PermissionError(f"Permission denied accessing directory {root_dir}: {e}")
        except OSError as e:
            raise OSError(f"Error reading directory {root_dir}: {e}")
        
        # Validate that we found images
        if not self.img_list:
            logger.warning(f"No valid image files found in {root_dir}")
        else:
            logger.info(f"Found {len(self.img_list)} valid image files in {root_dir}")
        
        # Initialize default transform if none provided
        self.default_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # Validate index
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
        
        if idx < 0 or idx >= len(self.img_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.img_list)}")
        
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            # Check if file still exists (in case it was deleted after initialization)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Load and convert image
            img = Image.open(img_path)
            
            # Verify the image can be loaded properly
            img.verify()  # This checks if the image is corrupted
            
            # Reopen the image since verify() closes it
            img = Image.open(img_path).convert('RGB')
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {img_path}")
            raise e
        except Image.UnidentifiedImageError as e:
            logger.error(f"Cannot identify image file: {img_path}")
            raise ValueError(f"Invalid or corrupted image file: {img_path}") from e
        except PermissionError as e:
            logger.error(f"Permission denied accessing file: {img_path}")
            raise e
        except OSError as e:
            logger.error(f"OS error loading image {img_path}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error loading image {img_path}: {e}")
            raise RuntimeError(f"Failed to load image {img_path}: {e}") from e
        
        # Apply transforms
        try:
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = self.default_transform(img)
                
        except Exception as e:
            logger.error(f"Error applying transforms to {img_path}: {e}")
            raise RuntimeError(f"Transform failed for image {img_path}: {e}") from e
        
        return img_tensor
    
    def get_image_info(self, idx):
        """Get information about an image without loading it"""
        if idx < 0 or idx >= len(self.img_list):
            raise IndexError(f"Index {idx} out of range")
        
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            with Image.open(img_path) as img:
                return {
                    'filename': img_name,
                    'path': img_path,
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format
                }
        except Exception as e:
            logger.error(f"Error getting info for {img_path}: {e}")
            return None