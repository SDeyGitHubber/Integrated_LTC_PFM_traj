import torch
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset
from utils.collate import collate_fn

from models.pfm_model_neighbours import PFMOnlyModel
from models.deep_lstm_model import DeepLSTMModel
from models.mta_pfm_model import CheckpointedIntegratedMTAPFMModel  # assuming this is your integrated model

from train.train_pfm import train_pfm_model as train_pfm_only_model
from train.train_deep_lstm import train_mta_model as train_deep_lstm_model
from train.train_pfm import train_pfm_model as train_integrated_pfm_model  # reuse train_pfm_model for integrated PFM


def main():
    print("Script started")
    data_path = "data/combined_annotations.csv"
    batch_size = 32
    epochs = 80
    learning_rate = 0.001
    patience = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting training for PFMOnlyModel...")
    train_pfm_only_model(
        data_path=data_path,
        model_save_path="checkpoints/pfm_only_model.pth",
        model_class=PFMOnlyModel,
        dataset_class=PFM_TrajectoryDataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
    )

    print("\nStarting training for DeepLSTMModel...")
    train_deep_lstm_model(
        data_path=data_path,
        model_save_path="checkpoints/deep_lstm_model.pth",
        model_class=DeepLSTMModel,
        dataset_class=PFM_TrajectoryDataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
    )

    print("\nStarting training for CheckpointedIntegratedMTAPFMModel...")
    train_integrated_pfm_model(
        data_path=data_path,
        model_save_path="checkpoints/integrated_pfm_model.pth",
        model_class=CheckpointedIntegratedMTAPFMModel,
        dataset_class=PFM_TrajectoryDataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
    )

    print("\nAll models trained successfully!")

if __name__ == "__main__":
    main()