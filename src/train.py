import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.callbacks import RichModelSummary
from datamodules.dogbreed_datamodule import DogBreedDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper
import os

def main():
    # Set up paths for Docker volumes
    data_dir = '/app/data'
    checkpoints_dir = '/app/checkpoints'
    logs_dir = '/app/logs'

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Set up logger
    logger = setup_logger(os.path.join(logs_dir, "train_log.log"))
    logger.info("Starting training...")

    # Initialize DataModule
    datamodule = DogBreedDataModule(data_dir=data_dir)
    
    # Initialize model
    model = DogBreedClassifier(num_classes=10) 
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='dog_breed_classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        period=1
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ],
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger('logs/', name='dog_breed_classifier')
    )
    
    # Train the model
    @task_wrapper
    trainer.fit(model, datamodule=datamodule)
    
    # Save the model
    final_model_path = os.path.join(checkpoints_dir, "dog_breed_classifier_final.ckpt")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Training completed. Final model saved as {final_model_path}")

if __name__ == "__main__":
    main()