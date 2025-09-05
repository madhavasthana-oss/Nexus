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

class UnlabelledDataset(Dataset):
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

class LabelledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for labelled images where each subdirectory in root_dir represents a class label,
        and images are contained within those subdirectories.
        
        Args:
            root_dir (str): Root directory containing subdirectories (class labels) with images.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = ('.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.tif')
        self.img_list = []
        self.labels = []
        self.label_to_idx = {}
        
        # Validate root directory
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
        
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"Path is not a directory: {root_dir}")
        
        # Load images and labels from directory structure
        self._load_from_directory_structure()
        
        # Validate that we found images
        if not self.img_list:
            logger.warning(f"No valid image files found in {root_dir}")
        else:
            logger.info(f"Found {len(self.img_list)} valid image files with labels in {root_dir}")
        
        # Initialize default transform if none provided
        self.default_transform = transforms.ToTensor()
    
    def _load_from_directory_structure(self):
        """Load images and labels from directory structure where subdirs are class names."""
        try:
            for subdir in sorted(os.listdir(self.root_dir)):
                subdir_path = os.path.join(self.root_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                
                # Assign label index
                if subdir not in self.label_to_idx:
                    self.label_to_idx[subdir] = len(self.label_to_idx)
                
                # Get valid image files in subdirectory
                for img_name in os.listdir(subdir_path):
                    if img_name.lower().endswith(self.valid_extensions):
                        self.img_list.append(os.path.join(subdir, img_name))
                        self.labels.append(self.label_to_idx[subdir])
        except PermissionError as e:
            raise PermissionError(f"Permission denied accessing directory {self.root_dir}: {e}")
        except OSError as e:
            raise OSError(f"Error reading directory {self.root_dir}: {e}")
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        """Get image and label by index."""
        # Validate index
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
        
        if idx < 0 or idx >= len(self.img_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.img_list)}")
        
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label = self.labels[idx]
        
        try:
            # Check if file still exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Load and convert image
            img = Image.open(img_path)
            
            # Verify the image
            img.verify()
            
            # Reopen since verify() closes the image
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
        
        return img_tensor, label
    
    def get_image_info(self, idx):
        """Get information about an image without loading it."""
        if idx < 0 or idx >= len(self.img_list):
            raise IndexError(f"Index {idx} out of range")
        
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label = self.labels[idx]
        label_name = next(k for k, v in self.label_to_idx.items() if v == label)
        
        try:
            with Image.open(img_path) as img:
                return {
                    'filename': img_name,
                    'path': img_path,
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'label': label_name,
                    'label_idx': label
                }
        except Exception as e:
            logger.error(f"Error getting info for {img_path}: {e}")
            return None
    
    def get_class_names(self):
        """Return dictionary mapping label indices to class names."""
        return {v: k for k, v in self.label_to_idx.items()}