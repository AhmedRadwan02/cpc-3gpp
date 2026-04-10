"""Data loading utilities for CSI datasets."""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class DatasetLoader:
    """Loader for CSI datasets from multiple vendors."""
    
    def __init__(self, main_path: str, window: Optional[int] = None):
        """
        Args:
            main_path: Base path to dataset directory
            window: Window size for temporal grouping (None for no windowing)
        """
        self.main_path = Path(main_path)
        self.window = window
        
        # Adjust batch size based on number of GPUs
        self.base_batch_size = 256
        self.num_gpus = torch.cuda.device_count()
        self.batch_size = self.base_batch_size * self.num_gpus if self.num_gpus > 0 else self.base_batch_size
        
    def _load_and_split_data(self, 
                            data_path: str, 
                            num_samples: Optional[int] = None, 
                            batch_size: Optional[int] = None,
                            dataset_name: str = "",
                            train_split: float = 0.8,
                            val_split: float = 0.1,
                            test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Helper method to load and split data.
        
        Args:
            data_path: Relative path to .npy file
            num_samples: Number of samples to load (None for all)
            batch_size: Batch size (None uses default)
            dataset_name: Name for logging
            train_split: Training split ratio
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        print(f"Loading {dataset_name} dataset...", flush=True)
        data = np.load(self.main_path / data_path)
        
        if num_samples is not None:
            data = data[:num_samples]
        print(f"{dataset_name} dataset shape: {data.shape}", flush=True)
    
        # If window is specified, reshape the data before converting to tensor
        if self.window is not None:
            n_samples = len(data)
            remainder = n_samples % self.window
            if remainder != 0:
                print(f"Warning: Dropping last {remainder} samples to fit window size {self.window}.", flush=True)
                data = data[:-remainder]
            # Reshape using numpy first
            data = data.reshape(-1, self.window, 2, 13, 32)
        
        # Convert to torch tensor
        data = torch.from_numpy(data).float()
        
        # Split data
        total_samples = len(data)
        train_size = int(train_split * total_samples)
        val_size = int(val_split * total_samples)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        print(f"Dataset splits - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}", flush=True)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def load_nokia_data(self, data_path: str, num_samples: Optional[int] = None, 
                       batch_size: int = 256, train_split: float = 0.8,
                       val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load Nokia dataset."""
        return self._load_and_split_data(data_path, num_samples, batch_size, "NOKIA",
                                         train_split, val_split)
    
    def load_oppo_data(self, data_path: str, num_samples: Optional[int] = None, 
                      batch_size: int = 256, train_split: float = 0.8,
                      val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load OPPO dataset."""
        return self._load_and_split_data(data_path, num_samples, batch_size, "OPPO",
                                         train_split, val_split)
    
    def load_cat_data(self, data_path: str, num_samples: Optional[int] = None, 
                     batch_size: int = 256, train_split: float = 0.8,
                     val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load CAT dataset."""
        return self._load_and_split_data(data_path, num_samples, batch_size, "CAT",
                                         train_split, val_split)
    
    def load_mixed_data(self, 
                       nokia_path: str,
                       oppo_path: str,
                       cat_path: str,
                       samples_per_dataset: int = 100000,
                       batch_size: int = 256,
                       train_split: float = 0.8,
                       val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and mix data from all three datasets.
        
        Args:
            nokia_path: Path to Nokia dataset
            oppo_path: Path to OPPO dataset
            cat_path: Path to CAT dataset
            samples_per_dataset: Number of samples to use from each dataset
            batch_size: Batch size
            train_split: Training split ratio
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        print("Creating mixed dataset...", flush=True)
        
        # Load the datasets from file
        nokia_data = np.load(self.main_path / nokia_path)
        oppo_data = np.load(self.main_path / oppo_path)
        cat_data = np.load(self.main_path / cat_path)
        
        # Use minimum available samples
        n_samples_nokia = len(nokia_data)
        n_samples_oppo = len(oppo_data)
        n_samples_cat = len(cat_data)
        
        min_samples = min(n_samples_nokia, n_samples_oppo, n_samples_cat)
        print(f"Using {min_samples} samples from each dataset (smallest available).", flush=True)
        
        nokia_data = nokia_data[:min_samples]
        oppo_data = oppo_data[:min_samples]
        cat_data = cat_data[:min_samples]
        
        # Apply windowing if specified
        if self.window is not None:
            remainder = min_samples % self.window
            if remainder != 0:
                print(f"Warning: Dropping last {remainder} samples to fit window size {self.window}.", flush=True)
                nokia_data = nokia_data[:-remainder]
                oppo_data = oppo_data[:-remainder]
                cat_data = cat_data[:-remainder]
            # Reshape to add time dimension
            nokia_data = nokia_data.reshape(-1, self.window, 2, 13, 32)
            oppo_data = oppo_data.reshape(-1, self.window, 2, 13, 32)
            cat_data = cat_data.reshape(-1, self.window, 2, 13, 32)
        
        # Convert to torch tensors
        nokia_data = torch.from_numpy(nokia_data).float()
        oppo_data = torch.from_numpy(oppo_data).float()
        cat_data = torch.from_numpy(cat_data).float()
        
        # Concatenate all datasets
        mixed_data = torch.cat([nokia_data, oppo_data, cat_data], dim=0)
        
        # Shuffle the combined dataset
        indices = torch.randperm(len(mixed_data))
        mixed_data = mixed_data[indices]
        
        # Split the data
        total_samples = len(mixed_data)
        train_size = int(train_split * total_samples)
        val_size = int(val_split * total_samples)
        
        train_data = mixed_data[:train_size]
        val_data = mixed_data[train_size:train_size + val_size]
        test_data = mixed_data[train_size + val_size:]
        
        print(f"Mixed dataset shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}", flush=True)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size)
        
        return train_loader, val_loader, test_loader


def load_all_datasets(data_path: str, 
                     datasets_config: List[Dict],
                     window_size: Optional[int] = None,
                     mixed_samples: int = 100000,
                     batch_size: int = 256,
                     train_split: float = 0.8,
                     val_split: float = 0.1) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
    """Load all configured datasets.
    
    Args:
        data_path: Base path to datasets
        datasets_config: List of dataset configurations with 'name', 'path', 'num_samples'
        window_size: Window size for temporal grouping
        mixed_samples: Samples per dataset for mixed dataset
        batch_size: Batch size
        train_split: Training split ratio
        val_split: Validation split ratio
        
    Returns:
        Dictionary mapping dataset names to (train, val, test) loaders
    """
    loader = DatasetLoader(main_path=data_path, window=window_size)
    datasets = {}
    
    # Load individual datasets
    for ds_config in datasets_config:
        name = ds_config['name']
        path = ds_config['path']
        num_samples = ds_config['num_samples']
        
        print(f"\nLoading {name} dataset...", flush=True)
        load_fn = getattr(loader, f'load_{name.lower()}_data')
        datasets[name] = load_fn(path, num_samples=num_samples, batch_size=batch_size,
                                train_split=train_split, val_split=val_split)
    
    # Load mixed dataset if we have multiple datasets
    if len(datasets_config) >= 3:
        print("\nLoading Mixed dataset...", flush=True)
        # Assume Nokia, OPPO, CAT order
        nokia_cfg = next(d for d in datasets_config if d['name'] == 'NOKIA')
        oppo_cfg = next(d for d in datasets_config if d['name'] == 'OPPO')
        cat_cfg = next(d for d in datasets_config if d['name'] == 'CAT')
        
        datasets['Mixed'] = loader.load_mixed_data(
            nokia_path=nokia_cfg['path'],
            oppo_path=oppo_cfg['path'],
            cat_path=cat_cfg['path'],
            samples_per_dataset=mixed_samples,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split
        )
    
    return datasets
