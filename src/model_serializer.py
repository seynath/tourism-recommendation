"""Model serialization and persistence with compression."""

import pickle
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime


class ModelSerializer:
    """Handles model persistence with compression."""
    
    @staticmethod
    def save(model: Any, path: str, metadata: Dict[str, Any]) -> None:
        """
        Serialize model with metadata to compressed format.
        
        This method saves a trained model to disk using pickle serialization
        with gzip compression. Metadata is stored alongside the model to track
        version, training date, and performance metrics.
        
        Args:
            model: The model object to serialize (any picklable object)
            path: File path where the model should be saved
            metadata: Dictionary containing model metadata:
                - 'version': Model version string
                - 'training_date': ISO format date string
                - 'metrics': Dictionary of performance metrics
                - Additional custom fields as needed
                
        Raises:
            ValueError: If path is empty or metadata is None
            IOError: If file cannot be written
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        if metadata is None:
            raise ValueError("Metadata cannot be None")
        
        # Ensure path has .pkl.gz extension
        path_obj = Path(path)
        if not path_obj.suffix == '.gz':
            if path_obj.suffix != '.pkl':
                path = str(path_obj) + '.pkl.gz'
            else:
                path = str(path_obj) + '.gz'
        
        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp if not present in metadata
        if 'saved_at' not in metadata:
            metadata['saved_at'] = datetime.now().isoformat()
        
        # Create serialization package
        package = {
            'model': model,
            'metadata': metadata
        }
        
        # Serialize with compression
        with gzip.open(path, 'wb', compresslevel=9) as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Deserialize model and return with metadata.
        
        This method loads a previously saved model from disk, decompressing
        and deserializing it along with its metadata.
        
        Args:
            path: File path to the saved model
            
        Returns:
            Tuple of (model, metadata) where:
                - model: The deserialized model object
                - metadata: Dictionary containing model metadata
                
        Raises:
            ValueError: If path is empty
            FileNotFoundError: If file does not exist
            IOError: If file cannot be read or is corrupted
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load and decompress
        try:
            with gzip.open(path, 'rb') as f:
                package = pickle.load(f)
            
            # Extract model and metadata
            model = package['model']
            metadata = package['metadata']
            
            return model, metadata
            
        except (pickle.UnpicklingError, gzip.BadGzipFile, KeyError) as e:
            raise IOError(f"Failed to load model from {path}: {str(e)}")
    
    @staticmethod
    def get_model_size(model: Any) -> float:
        """
        Calculate model size in MB.
        
        This method estimates the size of a model object by serializing it
        to bytes and measuring the result. This gives an accurate representation
        of the model's memory footprint.
        
        Args:
            model: The model object to measure
            
        Returns:
            Model size in megabytes (MB)
        """
        # Serialize to bytes to get accurate size
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        size_bytes = len(model_bytes)
        
        # Convert to MB
        return size_bytes / (1024 * 1024)
    
    @staticmethod
    def get_compressed_size(path: str) -> float:
        """
        Get the compressed file size in MB.
        
        Args:
            path: Path to the compressed model file
            
        Returns:
            File size in megabytes (MB)
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        size_bytes = path_obj.stat().st_size
        return size_bytes / (1024 * 1024)
    
    @staticmethod
    def save_metadata_only(metadata: Dict[str, Any], path: str) -> None:
        """
        Save only metadata to a JSON file.
        
        This is useful for storing model information without the full model,
        such as for model registry or tracking purposes.
        
        Args:
            metadata: Dictionary containing metadata to save
            path: File path where metadata should be saved (will add .json if needed)
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Ensure .json extension
        path_obj = Path(path)
        if path_obj.suffix != '.json':
            path = str(path_obj) + '.json'
        
        # Create parent directories if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        if 'saved_at' not in metadata:
            metadata['saved_at'] = datetime.now().isoformat()
        
        # Save as JSON
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_metadata_only(path: str) -> Dict[str, Any]:
        """
        Load only metadata from a JSON file.
        
        Args:
            path: File path to the metadata JSON file
            
        Returns:
            Dictionary containing metadata
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
