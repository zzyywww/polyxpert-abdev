import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset #Dataset class of hungging face
from torch.utils.data import DataLoader
from tqdm import tqdm
from abdev_core import BaseModel
import json
from pathlib import Path

class PolyXpertModel(BaseModel):
    """PolyXpert is a fine-tuned ESM-2 model for predicting polyreactivity of antibody candidates using scFv sequence data.
    
    Key features：
        Requires only heavy/light chain Fv sequences (no structural data).
        
    Development environment:
        Python 3.9.16
        CUDA Version: 12.2
        transformers version：4.26.1
        torch version：1.13.1
    """
    POLYXPERT_MODEL_NAME = "zhouyw/PolyXpert" #PolyXpert has been uploaded to hugging face
    
    def __init__(self) -> None:
        """Initialize model with lazy loading."""
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        self.tokenizer = None
        self.polyxpert = None
        self.device = None

        
    def _load_model(self)-> None:
        """Lazy load the model and tokenizer."""
        if self.polyxpert is not None:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading PolyXpert model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.POLYXPERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.POLYXPERT_MODEL_NAME,num_labels=2).to(self.device)
        self.model.eval()
    
    def _create_dataset(self, df: pd.DataFrame, max_length: int = 512):
        """prepare dataset"""
        seqs_df = df.copy()
        seqs_df["vh_protein_sequence"] = seqs_df["vh_protein_sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
        seqs_df["vl_protein_sequence"] = seqs_df["vl_protein_sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
        seqs_df['vh_protein_sequence'] = seqs_df.apply(lambda row: " ".join(row["vh_protein_sequence"]), axis=1)
        seqs_df['vl_protein_sequence'] = seqs_df.apply(lambda row: " ".join(row["vl_protein_sequence"]), axis=1)

        tokenized = self.tokenizer(
            list(seqs_df['vh_protein_sequence']),
            list(seqs_df['vl_protein_sequence']), 
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        dataset = Dataset.from_dict(tokenized)
        dataset = dataset.with_format("torch")
        return dataset

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """No-op training - this baseline uses pre-trained model.
        
        Saves temp files to run_dir for consistency.
        
        Args:
            df: Training dataframe (not used)
            run_dir: Directory to save temp files
            seed: Random seed (not used)
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration for reference
        config = {
            "model_type": "polyxpert",
            "note": "Non-training baseline using pre-trained fine tuned model"
        }

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        
        print("=" * 60)
        print("PolyXpert Model Information")
        print("=" * 60)
        print("PolyXpert is a pre-trained fine tuned classification model for antibody polyreactivity prediction.")
        print("The training process has already been completed. The trained model can be found in https://huggingface.co/zhouyw/PolyXpert or https://github.com/zzyywww/PolyXpert")
        print("")
        print("Model Components:")
        print("The model outputs a ployractivity probability ranging from 0 to 1.")
        print("     probability above 0.5 indicate high ployractivity, while scores below 0.5 indicate low ployractivity.")
        print("")
        print("To use PolyXpert, simply call the predict() method with your antibody sequences.")
        print("=" * 60)
    
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Predict polyreactivity for given antibody sequences.
        
        Args:
            df: DataFrame containing sequences with columns:
                - vh_protein_sequence: Heavy chain Fv sequence
                - vl_protein_sequence: Light chain Fv sequence
                - Name: Sequence identifier
            run_dir: Directory to save results
            
        Returns:
            DataFrame with prediction results containing:
                - proba_0: Probability of class 0 (low polyreactivity)
                - proba_1: Probability of class 1 (high polyreactivity)
                - pred_label: Predicted class (0 or 1)
        """
        self._load_model()
        # Create dataset
        test_data = self._create_dataset(df)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, pin_memory=True)
        
        # Run predictions
        predictions = []      
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                proba = torch.softmax(logits, dim=1)
                pred_label = torch.argmax(proba, dim=1)
                
                for i in range(proba.size(0)):
                    predictions.append({
                        'proba_0': proba[i, 0].item(),
                        'proba_1': proba[i, 1].item(),
                        'pred_label': pred_label[i].item(),
                    })
        
        # Convert to DataFrame
        result_temp = pd.DataFrame(predictions)
              
        
        result = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        result["PR_CHO"] = result_temp['proba_1']
        return result