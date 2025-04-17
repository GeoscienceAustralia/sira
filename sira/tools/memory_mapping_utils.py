import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import psutil
import pickle
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class MemoryTracker:
    """Track memory usage and provide warnings when approaching limits"""

    @staticmethod
    def get_memory_usage():
        """Get current process memory usage in GB"""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 ** 3)
        return memory_gb

    @staticmethod
    def get_available_memory():
        """Get available system memory in GB"""
        memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        return memory_gb

    @staticmethod
    def get_total_memory():
        """Get total system physical memory in GB"""
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        return memory_gb

    @staticmethod
    def log_memory_usage(task_name="Current"):
        """Log current memory usage"""
        used_gb = MemoryTracker.get_memory_usage()
        available_gb = MemoryTracker.get_available_memory()
        total_gb = psutil.virtual_memory().total / (1024 ** 3)

        logger.info(
            f"Memory Usage - {task_name}: "
            f"{used_gb:.2f} GB used, "
            f"{available_gb:.2f} GB available, "
            f"{total_gb:.2f} GB total")

        return used_gb, available_gb

    @staticmethod
    def check_memory_limit(threshold_gb=100):
        """Check if memory usage exceeds threshold, return True if safe"""
        used_gb = MemoryTracker.get_memory_usage()
        if used_gb > threshold_gb:
            logger.warning(
                f"Memory usage warning: {used_gb:.2f} GB used "
                f"exceeds threshold of {threshold_gb:.2f} GB")
            return False
        return True


class MemoryMappedData:
    """
    Handle memory-mapped storage of large data structures for shared memory access
    """

    def __init__(self, temp_dir=None):
        """Initialize with temporary directory for memory-mapped files"""
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="sira_memmap_"))
        else:
            self.temp_dir = Path(temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)

        self.data_files = {}
        self.metadata_file = self.temp_dir / "metadata.pkl"
        self.metadata = {}
        logger.info(f"Memory mapped data directory: {self.temp_dir}")

    def store_dataframe(self, df, name):
        """
        Store a pandas DataFrame as memory-mapped files

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to store
        name : str
            Name for the stored data

        Returns
        -------
        dict
            Metadata for accessing the memory-mapped data
        """
        # Convert DataFrame to numpy array and store values
        values_file = self.temp_dir / f"{name}_values.npy"
        mmap_values = np.memmap(
            values_file, dtype=df.values.dtype,
            mode='w+', shape=df.values.shape)
        mmap_values[:] = df.values[:]
        mmap_values.flush()

        # Store index
        index_file = self.temp_dir / f"{name}_index.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump({
                'index': df.index,
                'columns': df.columns,
                'shape': df.shape,
                'dtypes': df.dtypes
            }, f)

        # Record file information
        self.data_files[name] = {
            'values_file': str(values_file),
            'index_file': str(index_file),
            'shape': df.shape,
            'dtype': str(df.values.dtype)
        }

        # Update metadata
        self.metadata[name] = {
            'type': 'dataframe',
            'shape': df.shape,
            'dtype': str(df.values.dtype)
        }

        # Save metadata
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        MemoryTracker.log_memory_usage(f"After storing DataFrame {name}")

        return self.data_files[name]

    def store_array(self, arr, name):
        """
        Store a numpy array as memory-mapped file

        Parameters
        ----------
        arr : numpy.ndarray
            Array to store
        name : str
            Name for the stored data

        Returns
        -------
        dict
            Metadata for accessing the memory-mapped data
        """
        array_file = self.temp_dir / f"{name}.npy"
        mmap_array = np.memmap(
            array_file, dtype=arr.dtype,
            mode='w+', shape=arr.shape)
        mmap_array[:] = arr[:]
        mmap_array.flush()

        # Record file information
        self.data_files[name] = {
            'file': str(array_file),
            'shape': arr.shape,
            'dtype': str(arr.dtype)
        }

        # Update metadata
        self.metadata[name] = {
            'type': 'array',
            'shape': arr.shape,
            'dtype': str(arr.dtype)
        }

        # Save metadata
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        MemoryTracker.log_memory_usage(f"After storing array {name}")

        return self.data_files[name]

    def store_dict(self, dict_obj, name):
        """Store a dictionary object"""
        dict_file = self.temp_dir / f"{name}.pkl"
        with open(dict_file, 'wb') as f:
            pickle.dump(dict_obj, f)

        self.data_files[name] = {
            'file': str(dict_file),
            'type': 'dict'
        }

        # Update metadata
        self.metadata[name] = {
            'type': 'dict'
        }

        # Save metadata
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        return self.data_files[name]

    @staticmethod
    def load_dataframe(data_info):
        """
        Load a pandas DataFrame from memory-mapped files

        Parameters
        ----------
        data_info : dict
            Metadata for the stored DataFrame

        Returns
        -------
        pandas.DataFrame
            Reconstructed DataFrame
        """
        # Load values from memory-mapped file (read-only)
        mmap_values = np.memmap(
            data_info['values_file'],
            dtype=np.dtype(data_info['dtype']),
            mode='r',
            shape=tuple(data_info['shape']))

        # Load index information
        with open(data_info['index_file'], 'rb') as f:
            index_info = pickle.load(f)

        # Reconstruct DataFrame
        df = pd.DataFrame(
            mmap_values,
            index=index_info['index'],
            columns=index_info['columns'])

        return df

    @staticmethod
    def load_array(data_info):
        """
        Load a numpy array from memory-mapped file

        Parameters
        ----------
        data_info : dict
            Metadata for the stored array

        Returns
        -------
        numpy.ndarray
            Memory-mapped array (read-only)
        """
        mmap_array = np.memmap(
            data_info['file'],
            dtype=np.dtype(data_info['dtype']),
            mode='r',
            shape=tuple(data_info['shape']))
        return mmap_array

    @staticmethod
    def load_dict(data_info):
        """Load a dictionary object"""
        with open(data_info['file'], 'rb') as f:
            return pickle.load(f)

    def get_info(self):
        """Get information about stored data"""
        return self.metadata

    def get_data_info(self, name):
        """Get metadata for a specific stored object"""
        if name in self.data_files:
            return self.data_files[name]
        else:
            logger.warning(f"Data '{name}' not found in memory-mapped storage")
            return None

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up memory-mapped data directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up memory-mapped data: {e}")

    @staticmethod
    def open_existing(temp_dir):
        """Open an existing memory-mapped data storage"""
        mmap_data = MemoryMappedData(temp_dir)

        # Load metadata
        metadata_file = mmap_data.temp_dir / "metadata.pkl"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                mmap_data.metadata = pickle.load(f)

            # Reconstruct data_files entries
            for name, info in mmap_data.metadata.items():
                if info['type'] == 'dataframe':
                    mmap_data.data_files[name] = {
                        'values_file': str(mmap_data.temp_dir / f"{name}_values.npy"),
                        'index_file': str(mmap_data.temp_dir / f"{name}_index.pkl"),
                        'shape': info['shape'],
                        'dtype': info['dtype']
                    }
                elif info['type'] == 'array':
                    mmap_data.data_files[name] = {
                        'file': str(mmap_data.temp_dir / f"{name}.npy"),
                        'shape': info['shape'],
                        'dtype': info['dtype']
                    }
                elif info['type'] == 'dict':
                    mmap_data.data_files[name] = {
                        'file': str(mmap_data.temp_dir / f"{name}.pkl"),
                        'type': 'dict'
                    }

        return mmap_data
