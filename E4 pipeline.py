#
# This script trains an ensemble LSTM-GRU model, saves its training history
# to a CSV file, plots the loss curves, and generates a confusion matrix
# to evaluate performance on the test set.
#
# Before running this script, you MUST install the required libraries:
# pip install pandas numpy scikit-learn matplotlib tqdm
#

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import csv
import io

# Define the columns to be used as features and labels.
FEATURE_COLUMNS = ['BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA']
APNEA_COLUMNS = ['Obstructive_Apnea', 'Central_Apnea', 'Hypopnea', 'Multiple_Events']

# --- GPU Check ---
print("Checking for GPU availability...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("No GPU found. Training will run on CPU.")
    print("Ensure you have a compatible NVIDIA GPU and the CUDA Toolkit installed.")
# --- End of GPU Check ---

class ApneaDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and preprocessing the sleep apnea data.
    """
    def __init__(self, file_paths, scaler):
        """
        Args:
            file_paths (list): A list of file paths for the dataset.
            scaler (StandardScaler): A pre-fit scaler to apply.
        """
        self.file_paths = file_paths
        self.scaler = scaler
        
        print("Pre-loading data into memory...")
        self.data_frames = []
        self.offsets = []
        total_samples = 0
        for path in tqdm(self.file_paths, desc="Loading files"):
            try:
                df = pd.read_csv(path)
                if not all(col in df.columns for col in FEATURE_COLUMNS):
                    print(f"Warning: Skipping file {path} due to missing required columns.")
                    continue
                
                self.data_frames.append(df)
                self.offsets.append(total_samples)
                total_samples += len(df)
            except Exception as e:
                print(f"Skipping file {path} due to error during loading: {e}")
                continue
        self._total_samples = total_samples
        print(f"Loaded {self._total_samples} total samples from {len(self.data_frames)} files.")
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self._total_samples

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index from the pre-loaded data.
        """
        file_idx = next(i for i, offset in enumerate(self.offsets) if idx < offset + len(self.data_frames[i]))
        df = self.data_frames[file_idx]
        local_idx = idx - self.offsets[file_idx]

        try:
            features_raw = df.loc[local_idx, FEATURE_COLUMNS].values
            labels_raw = df.loc[local_idx, APNEA_COLUMNS].any().astype(int)
            
            features_scaled = self.scaler.transform(features_raw.reshape(1, -1)).flatten()
            
            if not np.isfinite(features_scaled).all():
                print(f"Warning: Found NaN or Inf values in sample at index {idx}. Skipping.")
                return None, None
            
            feature = torch.tensor(features_scaled.reshape(1, 1, -1), dtype=torch.float32)
            label = torch.tensor(labels_raw, dtype=torch.float32)
            
            return feature, label

        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
            return None, None

def custom_collate_fn(batch):
    """
    Custom collate function to handle `None` values from the dataset,
    which can happen when a sample is skipped due to an error.
    """
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    features = torch.cat([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return features, labels

class EnsembleModel(nn.Module):
    """
    An ensemble deep learning model with parallel LSTM and GRU branches.
    """
    def __init__(self, input_size, hidden_size=32):
        """
        Args:
            input_size (int): The number of features in each timestep.
            hidden_size (int): The number of units in the hidden layers.
        """
        super(EnsembleModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        """Defines the forward pass of the model."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        merged = torch.cat((lstm_out, gru_out), dim=1)
        output = self.fc(merged)
        return output

def plot_loss_curves(train_losses, val_losses):
    """
    Plots the training and validation loss curves.

    Args:
        train_losses (list): A list of training loss values per epoch.
        val_losses (list): A list of validation loss values per epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def evaluate_and_plot_confusion_matrix(model, test_loader, device):
    """
    Evaluates the model on the test set and plots a confusion matrix.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): The DataLoader for the test dataset.
        device (str): The device ('cuda' or 'cpu') to run the evaluation on.
    """
    print("\nEvaluating model on the test set...")
    model.eval() # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            if features.nelement() == 0:
                continue
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            # Apply sigmoid to convert logits to probabilities, then threshold to get binary predictions.
            preds = (torch.sigmoid(outputs) > 0.5).int().view(-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate the confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)

    # Display the confusion matrix with proper labels.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Apnea', 'Apnea'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Print a classification report for more detailed metrics.
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Apnea', 'Apnea']))

# Main execution block.
if __name__ == '__main__':
    try:
        data_path = r'E:\Muttakee\Polysomnography\DREAMT\DREAMT\data_64Hz'
        
        if not os.path.exists(data_path):
            print(f"Error: Data path not found at {data_path}. Please ensure the directory exists.")
        else:
            BATCH_SIZE = 48

            all_files = glob.glob(os.path.join(data_path, '*.csv'))
            if not all_files:
                raise ValueError("No CSV files found in the specified directory.")
            
            selected_files = all_files[:40]
            if len(selected_files) < 40:
                print(f"Warning: Only {len(selected_files)} files found. Proceeding with available files.")
            
            train_files, temp_files = train_test_split(selected_files, test_size=10, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=5, random_state=42)

            print(f"Dataset split: Training ({len(train_files)} files), Validation ({len(val_files)} files), Test ({len(test_files)} files)")

            print("Fitting scaler on a sample of data...")
            sample_files = np.random.choice(train_files, size=min(5, len(train_files)), replace=False)
            temp_df = pd.concat([pd.read_csv(f) for f in sample_files])
            temp_features = temp_df[FEATURE_COLUMNS].values
            scaler = StandardScaler()
            scaler.fit(temp_features)
            
            train_dataset = ApneaDataset(train_files, scaler)
            val_dataset = ApneaDataset(val_files, scaler)
            test_dataset = ApneaDataset(test_files, scaler)

            print("Creating DataLoaders with multiple workers...")
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            input_size = len(FEATURE_COLUMNS)
            model = EnsembleModel(input_size=input_size).to(device)
            
            print("Ensemble model architecture:")
            print(model)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # --- Training Loop with Loss Tracking ---
            print("Training the model...")
            num_epochs = 2
            train_losses = []
            val_losses = []
            
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
                    if features.nelement() == 0:
                        continue
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(features)
                    outputs = outputs.view(-1)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * features.size(0)
                
                if len(train_dataset) > 0:
                    epoch_loss = running_loss / len(train_dataset)
                    train_losses.append(epoch_loss)
                    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

                # --- Validation Loop ---
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                        if features.nelement() == 0:
                            continue
                        features = features.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(features)
                        outputs = outputs.view(-1)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * features.size(0)
                
                if len(val_dataset) > 0:
                    val_epoch_loss = val_loss / len(val_dataset)
                    val_losses.append(val_epoch_loss)
                    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}")

            print("Model training complete.")
            
            # --- Save the trained model ---
            model_dir = r'E:\Muttakee\Polysomnography\DREAMT'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, 'apnea_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # --- Save the loss history to CSV ---
            history_df = pd.DataFrame({
                'train_loss': train_losses,
                'val_loss': val_losses
            })
            csv_path = os.path.join(model_dir, 'training_history.csv')
            history_df.to_csv(csv_path, index=False)
            print(f"Training history saved to {csv_path}")

            # --- Step 1: Plot the loss curves ---
            plot_loss_curves(train_losses, val_losses)
            
            # --- Step 2: Evaluate on the test set and plot the confusion matrix ---
            evaluate_and_plot_confusion_matrix(model, test_loader, device)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
