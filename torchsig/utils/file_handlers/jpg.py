from __future__ import annotations

# TorchSig
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.datasets.dataset_metadata import DatasetMetadata

# Third Party
from PIL import Image  # Explicitly import Image from PIL
import numpy as np

# Built-In
from typing import TYPE_CHECKING, Tuple, List, Dict, Any
import os
import pickle

if TYPE_CHECKING:
    from torchsig.datasets.datasets import NewTorchSigDataset

class JPGFileHandler(TorchSigFileHandler):
    """Handler for reading and writing data to/from a JPG file format.

    This class extends the `TorchSigFileHandler` and provides functionality to handle 
    reading, writing, and managing JPG-based storage for dataset samples.

    Attributes:
        datapath_filename (str): The name of the file used to store the data in jpg format.
    """

    image_file_prefix = 'idx'
    image_file_extention = '.jpg'
    chunk_size = (100, )

    def __init__(
        self,
        root: str,
        dataset_metadata: DatasetMetadata,
        batch_size: int,
        train: bool = None,
    ):
        """Initializes the ZarrFileHandler with dataset metadata and write type.

        Args:
            dataset_metadata (DatasetMetadata): Metadata about the dataset, including 
                sample sizes and other configuration.
            write_type (str, optional): Specifies the write mode for the dataset ("raw" or otherwise). 
                Defaults to None.
        """
        super().__init__(
            root = root,
            dataset_metadata = dataset_metadata,
            batch_size = batch_size,
            train = train,
        )

        self.datapath = f"{self.root}"
        self.targets_pickle_filename='targets.pkl'
        self.data_shape = (self.dataset_metadata.num_samples, self.dataset_metadata.num_iq_samples_dataset)
        self.data_type = float


    def exists(self) -> bool:
        """Checks if the dataset directory exists at the specified path.

        Returns:
            bool: True if the dataset directory exists, otherwise False.
        """
        if os.path.exists(self.datapath):
            return True
        else:
            return False
        

    def write(self, batch_idx: int, batch: Any) -> None:
        """Writes a sample (data and targets) to the JPG file at the specified index.

        Args:
            idx (int): The index at which to store the data in the dataset directory.
            data (np.ndarray): The data to write to the JPG image file.
            targets (Any): The corresponding targets to write as metadata for the sample.
        
        Notes:
            If the index is greater than the current size of the array, the array is 
            expanded to accommodate the new sample.
        """

        start_idx = batch_idx * self.batch_size
        stop_idx = start_idx + len(batch[0])

        data, targets = batch

        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        elif os.path.exists(self.datapath+"/"+self.targets_pickle_filename):
            with open(self.datapath+"/"+self.targets_pickle_filename, "rb") as pickle_file:
                metadata_dict = pickle.load(pickle_file)
        else:
            metadata_dict=dict()

        for idx,(data,targets_metdada) in enumerate(zip(data,targets),start=start_idx):
            image_file_name=self.image_file_prefix+'_'+str(idx)+self.image_file_extention
            image=Image.fromarray((data * 255).astype(np.uint8))  # Use Image from PIL
            try:
                image.save(self.datapath+"/"+image_file_name,'JPEG')
                with open(self.datapath+"/"+self.targets_pickle_filename, 'wb') as handle:
                    metadata_dict[idx]=targets_metdada
                    pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                print('Failed to save dataset')
                print(e)
   
    def load(self,idx:int)->Tuple[np.ndarray, Dict[str, Any]] | Tuple[Any,...]:

        if not os.path.exists(self.datapath):
            print(f'Path {os.path.exists(self.datapath)} does not exist')
        elif os.path.exists(self.datapath+"/"+self.targets_pickle_filename):
            with open(self.datapath+"/"+self.targets_pickle_filename, "rb") as pickle_file:
                metadata_dict = pickle.load(pickle_file)
        else:
            metadata_dict=dict()

        if idx in metadata_dict.keys():
            image_file_name=self.image_file_prefix+'_'+str(idx)+self.image_file_extention
            image_file_path=str(self.datapath+'/'+image_file_name)
            if not os.path.exists(image_file_path):
                print(f'could not load filename: {image_file_path}')
                return None
            try:
                spec_image=Image.open(image_file_path)  # Use Image from PIL

            except Exception as e:
                print(f'could not load image: {image_file_path}')
                return None

        return spec_image,metadata_dict
