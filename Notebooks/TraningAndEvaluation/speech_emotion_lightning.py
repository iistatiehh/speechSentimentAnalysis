import torch
import os
import pandas as pd
import torchaudio
import lightning as L
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

class SpeechEmotionDataModule(L.LightningDataModule):
    def __init__(self, data_dir="data", processed_data_dir="processed_data", batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.processed_data_dir = processed_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sampling_rate = 16000
        
        # Initialize processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
        # These will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label2id = None
        self.id2label = None
        
    def prepare_data(self):
        """Download and prepare data (called only once)"""
        print("🔄 Preparing data...")
        
        # Check if processed data exists and is complete
        if self._is_processed_data_complete():
            print(f"✅ Processed data found at {self.processed_data_dir}")
            return
        
        # Load and process the original data
        print("🔄 Processing raw data (processed data incomplete or missing)...")
        self._load_and_process_raw_data()
    
    def _is_processed_data_complete(self):
        """Check if all required processed files exist"""
        if not os.path.exists(self.processed_data_dir):
            return False
        
        required_files = [
            "train_processed.pkl",
            "valid_processed.pkl", 
            "test_processed.pkl",
            "label_mappings.json"
        ]
        
        for file_name in required_files:
            file_path = os.path.join(self.processed_data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"❌ Missing required file: {file_path}")
                return False
        
        return True
        
    def _load_and_process_raw_data(self):
        """Load and process raw data from CSV"""
        print("📁 Loading raw data...")
        
        # Load the CSV file
        df = pd.read_csv(os.path.join(self.data_dir, "afterReadingDataSet.csv"))
        print(f"Original DataFrame shape: {df.shape}")
        
        # Fix column names
        df = df.rename(columns={'Emotions': 'label', 'Path': 'path'})
        
        # Remove rows with null path or label
        df = df.dropna(subset=["path", "label"])
        print(f"DataFrame shape after cleaning: {df.shape}")
        
        # Remove rows where path file doesn't exist
        df = df[df["path"].apply(os.path.exists)]
        print(f"DataFrame shape after file existence check: {df.shape}")
        
        # Map emotion labels to integers
        label_list = sorted(df["label"].unique())
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        print(f"Found {len(label_list)} unique labels: {label_list}")
        
        # Add label_id column
        df["label_id"] = df["label"].map(self.label2id)
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        
        # Split dataset
        train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
        valid_test = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        
        train_dataset = train_testvalid['train']
        valid_dataset = valid_test['train']
        test_dataset = valid_test['test']
        
        print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
        
        # Process datasets
        print("🔄 Processing datasets...")
        train_processed = self._process_dataset(train_dataset, "train")
        valid_processed = self._process_dataset(valid_dataset, "valid")
        test_processed = self._process_dataset(test_dataset, "test")
        
        # Save processed data
        self._save_processed_data(train_processed, valid_processed, test_processed)
        
    def _process_dataset(self, dataset, split_name):
        """Process a dataset split"""
        processed = []
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                print(f"Processing {split_name} example {i}/{len(dataset)}")
            
            processed_example = self._preprocess_example(example)
            if processed_example is not None:
                processed.append(processed_example)
        
        print(f"Successfully processed {len(processed)}/{len(dataset)} {split_name} examples")
        return processed
    
    def _preprocess_example(self, example):
        """Preprocess a single audio example"""
        try:
            # Load audio file
            speech_array, sampling_rate = torchaudio.load(example['path'])
            
            # Resample if necessary
            if sampling_rate != self.target_sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, 
                    new_freq=self.target_sampling_rate
                )
                speech_array = resampler(speech_array)
            
            # Convert to mono if stereo
            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)
            
            # Process with Wav2Vec2 processor
            inputs = self.processor(
                speech_array.squeeze().numpy(), 
                sampling_rate=self.target_sampling_rate, 
                return_tensors="pt"
            )
            
            # Extract input_values and convert to list for Arrow compatibility
            input_values = inputs["input_values"].squeeze().tolist()
            
            # Update example with processed data
            example["input_values"] = input_values
            example["labels"] = example["label_id"]
            
            return example
            
        except Exception as e:
            print(f"Error processing {example['path']}: {e}")
            return None
    
    def _save_processed_data(self, train_processed, valid_processed, test_processed):
        """Save processed data to disk"""
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Save as pickle files
        with open(os.path.join(self.processed_data_dir, "train_processed.pkl"), 'wb') as f:
            pickle.dump(train_processed, f)
        
        with open(os.path.join(self.processed_data_dir, "valid_processed.pkl"), 'wb') as f:
            pickle.dump(valid_processed, f)
        
        with open(os.path.join(self.processed_data_dir, "test_processed.pkl"), 'wb') as f:
            pickle.dump(test_processed, f)
        
        # Save label mappings
        mappings = {
            "label2id": self.label2id,
            "id2label": self.id2label,
            "label_list": list(self.label2id.keys()),
            "num_labels": len(self.label2id)
        }
        
        with open(os.path.join(self.processed_data_dir, "label_mappings.json"), 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"✅ Saved processed data to {self.processed_data_dir}")
    
    def setup(self, stage=None):
        """Setup datasets for training/validation/testing"""
        print(f"🔄 Setting up datasets for stage: {stage}")
        
        # Load label mappings
        mappings_path = os.path.join(self.processed_data_dir, "label_mappings.json")
        
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(
                f"Label mappings file not found at {mappings_path}. "
                f"Please delete the processed_data directory and run again to reprocess the data."
            )
        
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        self.label2id = mappings["label2id"]
        self.id2label = mappings["id2label"]
        self.num_labels = mappings["num_labels"]
        
        # Load processed data
        if stage == "fit" or stage is None:
            train_path = os.path.join(self.processed_data_dir, "train_processed.pkl")
            valid_path = os.path.join(self.processed_data_dir, "valid_processed.pkl")
            
            if not os.path.exists(train_path) or not os.path.exists(valid_path):
                raise FileNotFoundError(
                    f"Training/validation data files not found. "
                    f"Please delete the processed_data directory and run again to reprocess the data."
                )
            
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(valid_path, 'rb') as f:
                valid_data = pickle.load(f)
            
            self.train_dataset = Dataset.from_list(train_data)
            self.val_dataset = Dataset.from_list(valid_data)
            
        if stage == "test" or stage is None:
            test_path = os.path.join(self.processed_data_dir, "test_processed.pkl")
            
            if not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"Test data file not found at {test_path}. "
                    f"Please delete the processed_data directory and run again to reprocess the data."
                )
            
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)
            self.test_dataset = Dataset.from_list(test_data)
        
        print(f"✅ Datasets loaded successfully")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Extract input_values and labels
        input_values = [torch.tensor(example["input_values"]) for example in batch]
        labels = [example["labels"] for example in batch]
        
        # Pad sequences to the same length
        max_length = max(len(iv) for iv in input_values)
        padded_input_values = []
        
        for iv in input_values:
            if len(iv) < max_length:
                # Pad with zeros
                padded = torch.cat([iv, torch.zeros(max_length - len(iv))])
            else:
                padded = iv[:max_length]  # Truncate if too long
            padded_input_values.append(padded)
        
        return {
            "input_values": torch.stack(padded_input_values),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class SpeechEmotionClassifier(L.LightningModule):
    def __init__(self, num_labels, learning_rate=2e-5, model_name="facebook/wav2vec2-large-960h-lv60-self"):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        
        # Load pre-trained model
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_values"], batch["labels"])
        loss = outputs.loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_values"], batch["labels"])
        loss = outputs.loss
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=-1)
        
        # Store predictions and labels for epoch-end metrics
        self.validation_step_outputs.append({
            "loss": loss,
            "preds": preds,
            "labels": batch["labels"]
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute metrics for the entire validation set
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())
        f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
        
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        
        # Clear the outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_values"], batch["labels"])
        loss = outputs.loss
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=-1)
        
        self.test_step_outputs.append({
            "loss": loss,
            "preds": preds,
            "labels": batch["labels"]
        })
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        # Compute final test metrics
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())
        f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
        
        self.log("test_accuracy", accuracy, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)
        
        # Print detailed classification report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(all_labels.cpu(), all_preds.cpu()))
        
        # Clear the outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def main():
    """Main function to run the training"""
    print("🎯 Starting Speech Emotion Recognition Training")
    print("="*60)
    
    # Initialize data module
    data_module = SpeechEmotionDataModule(
        data_dir="../../data",  # Adjust path as needed
        processed_data_dir=r"C:\Users\asus\Desktop\SpeechSentemintAnalysis\processed_data",
        batch_size=8,  # Adjust based on your GPU memory
        num_workers=0  # Set to 0 for Windows
    )
    
    # Prepare data (this will process if needed)
    data_module.prepare_data()
    
    # Setup for training
    data_module.setup("fit")
    
    # Initialize model
    model = SpeechEmotionClassifier(
        num_labels=data_module.num_labels,
        learning_rate=2e-5
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=5,  # Reduced for quick testing
        accelerator="auto",  # Will use GPU if available
        devices="auto",
        precision="16-mixed",  # Updated precision format
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}"
            ),
            L.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            ),
            L.callbacks.LearningRateMonitor(logging_interval="step")
        ]
    )
    
    # Train the model
    print("🚀 Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    print("🧪 Testing the model...")
    data_module.setup("test")
    trainer.test(model, data_module)
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()