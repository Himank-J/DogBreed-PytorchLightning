import argparse
import json
from pathlib import Path
import torch
import pytorch_lightning as pl
from datamodules.dogbreed_datamodule import DogBreedDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper

@task_wrapper
def main(args):
    logger = setup_logger(__name__, log_file=Path(args.log_dir) / "test_log.log")
    logger.info("Starting evaluation...")
    
    # Initialize DataModule
    datamodule = DogBreedDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    datamodule.setup(stage='test')
    
    # Load the model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)
    
    # Initialize Trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        default_root_dir=args.output_dir
    )
    
    # Evaluate the model
    results = trainer.test(model, datamodule=datamodule)
    
    # Save results as JSON
    results_file = Path(args.output_dir) / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results[0], f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Dog Breed Classifier")
    parser.add_argument("--data_dir", type=str, default="/app/data", help="Path to data directory")
    parser.add_argument("--ckpt_path", type=str, default="/app/checkpoints/model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="/app/results", help="Path to output directory for results")
    parser.add_argument("--log_dir", type=str, default="/app/logs", help="Path to log directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    args = parser.parse_args()

    main(args)