import os
import getpass
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

class DogBreedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print('Done')
    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, 'dog-breed-image-dataset')):
            self._download_dataset()

    def _download_dataset(self):
        # Check if Kaggle credentials are set in environment variables
        if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
            print("Kaggle credentials not found in environment variables.")
            self._get_kaggle_credentials()

        # Import KaggleApi here to avoid immediate authentication
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Download the dataset using Kaggle API
        api = KaggleApi()
        try:
            api.authenticate()
            api.dataset_download_files('khushikhushikhushi/dog-breed-image-dataset', path=self.data_dir, unzip=True)
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure your Kaggle credentials are correct and you have internet connection.")
            raise

    def _get_kaggle_credentials(self):
        """Prompt user for Kaggle credentials and set them as environment variables."""
        print("Please enter your Kaggle credentials.")
        username = input("Kaggle username: ")
        key = getpass.getpass("Kaggle API key: ")
        
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        print("Credentials set for this session.")

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'dog-breed-image-dataset'), transform=self.transform)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)