"""
MTL-TABlock:  Model Training Main Program

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score, roc_auc_score
)
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
import warnings
from collections import Counter

from mtl_model import (
    MTLTABlockModel, MTLLoss, MTLLossWithLogits,
    SUBTYPE_LABELS, SUBTYPE_TO_IDX, IDX_TO_SUBTYPE,
    create_model, ModelConfig
)

# Configure logging
logging.basicConfig(
    level=logging. INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig: 
    """Training configuration matching paper Section 3.5.2."""
    # Data split ratios
    test_size: float = 0.2
    val_size: float = 0.2  # Of remaining after test split
    
    # Training hyperparameters
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularization
    
    # Loss weights (from paper:  λ_det > λ_type)
    lambda_det: float = 1.0
    lambda_type: float = 0.5
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Detection threshold
    detection_threshold: float = 0.5
    
    # Random seed
    random_state: int = 42
    
    # Class balancing
    use_class_weights:  bool = True
    use_oversampling: bool = False
    
    # Misc
    num_workers: int = 4
    pin_memory: bool = True
    
    def to_dict(self) -> Dict: 
        return asdict(self)


class FunctionFeatureDataset(Dataset):
    """
    Dataset for function-level features and labels.
    
    Each sample contains:
    - features: Combined structural and contextual features
    - tracking_label: Binary label (0=non-tracking, 1=tracking)
    - subtype_label:  Subtype label (0=benign, 1-4=tracking subtypes)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        tracking_labels: np.ndarray,
        subtype_labels: np. ndarray,
        script_ids: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset. 
        
        Args:
            features:  Feature matrix [N, D]
            tracking_labels: Binary tracking labels [N]
            subtype_labels: Subtype labels [N]
            script_ids: Optional script identifiers for grouping
        """
        self.features = torch.FloatTensor(features)
        self.tracking_labels = torch.FloatTensor(tracking_labels)
        self.subtype_labels = torch.LongTensor(subtype_labels)
        self.script_ids = script_ids
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        return (
            self. features[idx],
            self.tracking_labels[idx],
            self.subtype_labels[idx]
        )
    
    def get_class_distribution(self) -> Dict[str, Dict[int, int]]: 
        """Get distribution of classes."""
        tracking_dist = Counter(self. tracking_labels.numpy().astype(int))
        subtype_dist = Counter(self. subtype_labels. numpy().astype(int))
        return {
            "tracking": dict(tracking_dist),
            "subtype": dict(subtype_dist)
        }


class DataProcessor:
    """
    Processes raw data for model training.
    
    Implements data loading, preprocessing, and splitting as described
    in Section 3.5.2 of the paper. 
    """
    
    # Feature column names based on paper Table 2
    STRUCTURAL_FEATURES = [
        'num_nodes', 'num_edges', 'nodes_div_by_edges', 'edges_div_by_nodes',
        'in_degree', 'out_degree', 'in_out_degree', 'ancestor', 'descendants',
        'closeness_centrality', 'in_degree_centrality', 'out_degree_centrality',
        'is_anonymous', 'is_eval_or_external_function',
        'descendant_of_eval_or_function', 'ascendant_script_has_eval_or_function',
        'num_script_successors', 'num_script_predecessors',
        'num_method_successors', 'num_method_predecessors',
        'descendant_of_storage_node', 'ascendant_of_storage_node',
        'is_initiator', 'immediate_method',
        'descendant_of_fingerprinting', 'ascendant_of_fingerprinting'
    ]
    
    CONTEXTUAL_FEATURES = [
        'num_req_sent', 'storage_getter', 'storage_setter',
        'cookie_getter', 'cookie_setter',
        'getAttribute', 'setAttribute', 'addEventListener',
        'removeAttribute', 'removeEventListener', 'sendBeacon',
        'has_fingerprinting_eventlistner',
        'num_local', 'num_closure', 'num_global', 'num_script'
    ]
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd. read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df
    
    def load_from_excel(self, file_path: str) -> pd.DataFrame:
        """Load dataset from Excel file."""
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        subtype_col: str = 'subtype',
        script_col: str = 'script_name',
        exclude_unknown: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """
        Preprocess dataframe for training.
        
        Args:
            df: Input dataframe
            label_col:  Column name for tracking label
            subtype_col: Column name for subtype label
            script_col: Column name for script identifier
            exclude_unknown: Whether to exclude unknown subtypes (per paper Section 3.4. 2)
            
        Returns:
            Tuple of (features, tracking_labels, subtype_labels, script_ids)
        """
        # Remove duplicates by script and method (per existing code)
        if 'method_name' in df.columns and script_col in df. columns:
            df = df.drop_duplicates(
                subset=[script_col, 'method_name'],
                keep='last'
            ).reset_index(drop=True)
        
        # Exclude unknown subtypes (per paper Section 3.4.2)
        if exclude_unknown and subtype_col in df.columns:
            original_len = len(df)
            df = df[df[subtype_col] != 'unknown']
            df = df[df[subtype_col] != -1]  # Handle numeric unknown
            logger.info(f"Excluded {original_len - len(df)} unknown subtype samples")
        
        # Extract features
        feature_cols = self._get_feature_columns(df)
        self.feature_names = feature_cols
        features = df[feature_cols].values. astype(np. float32)
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract labels
        tracking_labels = df[label_col].values. astype(np.float32)
        
        # Convert subtype labels
        if subtype_col in df.columns:
            subtype_labels = self._convert_subtype_labels(df[subtype_col])
        else: 
            # Infer from tracking label:  tracking=1 gets subtype 1, non-tracking gets 0
            subtype_labels = tracking_labels.astype(np.int64)
        
        # Extract script IDs for stratified splitting
        script_ids = df[script_col]. values if script_col in df.columns else None
        
        logger.info(f"Preprocessed {len(features)} samples with {len(feature_cols)} features")
        
        return features, tracking_labels, subtype_labels, script_ids
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify feature columns in dataframe."""
        # Try to find known feature columns
        feature_cols = []
        
        # Check for structural features
        for col in self.STRUCTURAL_FEATURES: 
            if col in df.columns:
                feature_cols.append(col)
        
        # Check for contextual features
        for col in self. CONTEXTUAL_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        
        # If no known columns found, use Feature_N pattern
        if not feature_cols:
            feature_cols = [col for col in df.columns if col. startswith('Feature')]
        
        # If still no features, use all numeric columns except labels
        if not feature_cols:
            exclude_cols = {'label', 'subtype', 'script_name', 'method_name', 
                          'is_mixed', 'Unnamed: 0'}
            feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
        
        return feature_cols
    
    def _convert_subtype_labels(self, subtype_series: pd.Series) -> np.ndarray:
        """Convert subtype labels to numeric indices."""
        labels = np.zeros(len(subtype_series), dtype=np.int64)
        
        for i, val in enumerate(subtype_series):
            if isinstance(val, str):
                labels[i] = SUBTYPE_TO_IDX. get(val. lower(), 0)
            elif isinstance(val, (int, float)):
                labels[i] = int(val) if not np.isnan(val) else 0
            else: 
                labels[i] = 0
        
        return labels
    
    def split_by_script(
        self,
        features: np.ndarray,
        tracking_labels: np. ndarray,
        subtype_labels:  np.ndarray,
        script_ids: np.ndarray = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data by script granularity as described in Section 3.5.2.
        
        Ensures functions from the same script don't appear in both
        training and test sets.
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if script_ids is None:
            # Fall back to random split
            return self._random_split(features, tracking_labels, subtype_labels)
        
        # Get unique scripts
        unique_scripts = np. unique(script_ids)
        n_scripts = len(unique_scripts)
        
        # Create script-level labels for stratification
        script_labels = {}
        for script, label in zip(script_ids, tracking_labels):
            if script not in script_labels:
                script_labels[script] = []
            script_labels[script]. append(label)
        
        # Use majority label for each script
        script_majority_labels = {
            script: 1 if sum(labels) > len(labels) / 2 else 0
            for script, labels in script_labels. items()
        }
        
        # Split scripts
        script_array = np.array(list(script_majority_labels.keys()))
        label_array = np. array(list(script_majority_labels. values()))
        
        # First split:  train+val vs test
        scripts_trainval, scripts_test = train_test_split(
            script_array,
            test_size=self.config.test_size,
            random_state=self. config.random_state,
            stratify=label_array
        )
        
        # Second split:  train vs val
        trainval_labels = [script_majority_labels[s] for s in scripts_trainval]
        scripts_train, scripts_val = train_test_split(
            scripts_trainval,
            test_size=self.config. val_size / (1 - self.config.test_size),
            random_state=self. config.random_state,
            stratify=trainval_labels
        )
        
        # Create masks
        train_mask = np. isin(script_ids, scripts_train)
        val_mask = np.isin(script_ids, scripts_val)
        test_mask = np.isin(script_ids, scripts_test)
        
        logger.info(f"Split by script: {len(scripts_train)} train, "
                   f"{len(scripts_val)} val, {len(scripts_test)} test scripts")
        logger.info(f"Sample counts: {train_mask.sum()} train, "
                   f"{val_mask.sum()} val, {test_mask.sum()} test")
        
        return {
            'train': (features[train_mask], tracking_labels[train_mask], subtype_labels[train_mask]),
            'val': (features[val_mask], tracking_labels[val_mask], subtype_labels[val_mask]),
            'test': (features[test_mask], tracking_labels[test_mask], subtype_labels[test_mask])
        }
    
    def _random_split(
        self,
        features: np.ndarray,
        tracking_labels: np. ndarray,
        subtype_labels:  np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Random split when script IDs not available."""
        # First split
        X_trainval, X_test, y_det_trainval, y_det_test, y_type_trainval, y_type_test = \
            train_test_split(
                features, tracking_labels, subtype_labels,
                test_size=self.config. test_size,
                random_state=self.config.random_state,
                stratify=tracking_labels
            )
        
        # Second split
        X_train, X_val, y_det_train, y_det_val, y_type_train, y_type_val = \
            train_test_split(
                X_trainval, y_det_trainval, y_type_trainval,
                test_size=self. config.val_size / (1 - self.config.test_size),
                random_state=self.config.random_state,
                stratify=y_det_trainval
            )
        
        return {
            'train': (X_train, y_det_train, y_type_train),
            'val': (X_val, y_det_val, y_type_val),
            'test':  (X_test, y_det_test, y_type_test)
        }
    
    def fit_scaler(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler on training data and transform."""
        return self.scaler. fit_transform(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        return self.scaler.transform(features)


class EarlyStopping: 
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self. counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score:  float, model: nn.Module) -> bool:
        if self.best_score is None: 
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        if self.mode == 'max':
            improved = score > self. best_score + self.min_delta
        else: 
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = {k: v. cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else: 
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self. early_stop
    
    def load_best_model(self, model: nn.Module) -> None:
        """Load the best model state."""
        if self.best_model_state is not None:
            model. load_state_dict(self.best_model_state)


class MTLTrainer:
    """
    Trainer for MTL-TABlock model.
    
    Implements the training procedure described in Section 3.5.2.
    """
    
    def __init__(
        self,
        model: MTLTABlockModel,
        config: TrainingConfig = None,
        device: torch.device = None
    ):
        self.model = model
        self. config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda. is_available() else "cpu")
        
        self.model.to(self. device)
        
        # Initialize loss function
        self. criterion = MTLLoss(
            lambda_det=self.config.lambda_det,
            lambda_type=self.config.lambda_type
        )
        
        # Initialize optimizer (Adam per paper)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch. optim.lr_scheduler.ReduceLROnPlateau(
            self. optimizer,
            mode='max',
            factor=0.5,
            patience=self.config.patience // 2,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'train_det_loss':  [], 'train_type_loss': [],
            'val_loss': [], 'val_det_loss': [], 'val_type_loss':  [],
            'val_det_f1': [], 'val_type_f1_macro': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]: 
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        det_loss = 0
        type_loss = 0
        n_batches = 0
        
        for features, y_det, y_type in train_loader:
            features = features.to(self.device)
            y_det = y_det.to(self.device)
            y_type = y_type.to(self.device)
            
            self.optimizer.zero_grad()
            
            p_det, p_type = self.model(features)
            loss, losses = self.criterion(p_det, p_type, y_det, y_type)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn. utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer. step()
            
            total_loss += losses["loss_total"]
            det_loss += losses["loss_det"]
            type_loss += losses["loss_type"]
            n_batches += 1
        
        return {
            "loss_total": total_loss / n_batches,
            "loss_det":  det_loss / n_batches,
            "loss_type": type_loss / n_batches
        }
    
    def evaluate(
        self,
        data_loader: DataLoader,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation/test set.
        
        Returns metrics per paper Section 4.1:
        - Detection:  Precision, Recall, F1
        - Subtype: Per-class F1, Macro F1
        """
        if threshold is None: 
            threshold = self. config.detection_threshold
        
        self.model.eval()
        
        all_det_preds = []
        all_det_probs = []
        all_det_labels = []
        all_type_preds = []
        all_type_labels = []
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for features, y_det, y_type in data_loader: 
                features = features. to(self.device)
                y_det = y_det.to(self. device)
                y_type = y_type.to(self.device)
                
                p_det, p_type = self.model(features)
                loss, _ = self.criterion(p_det, p_type, y_det, y_type)
                
                total_loss += loss.item()
                n_batches += 1
                
                det_probs = p_det.squeeze().cpu().numpy()
                det_preds = (det_probs >= threshold).astype(int)
                type_preds = torch.argmax(p_type, dim=-1).cpu().numpy()
                
                all_det_preds.extend(det_preds)
                all_det_probs.extend(det_probs)
                all_det_labels.extend(y_det. cpu().numpy())
                all_type_preds.extend(type_preds)
                all_type_labels.extend(y_type.cpu().numpy())
        
        all_det_preds = np.array(all_det_preds)
        all_det_probs = np.array(all_det_probs)
        all_det_labels = np.array(all_det_labels)
        all_type_preds = np.array(all_type_preds)
        all_type_labels = np. array(all_type_labels)
        
        # Detection metrics
        det_precision = precision_score(all_det_labels, all_det_preds, zero_division=0)
        det_recall = recall_score(all_det_labels, all_det_preds, zero_division=0)
        det_f1 = f1_score(all_det_labels, all_det_preds, zero_division=0)
        
        try:
            det_auc = roc_auc_score(all_det_labels, all_det_probs)
        except ValueError:
            det_auc = 0.0
        
        # Subtype metrics (only for tracking functions per paper)
        tracking_mask = all_det_labels == 1
        if tracking_mask.sum() > 0:
            # Per-class F1 for tracking subtypes (classes 1-4)
            type_f1_per_class = f1_score(
                all_type_labels[tracking_mask],
                all_type_preds[tracking_mask],
                labels=[1, 2, 3, 4],
                average=None,
                zero_division=0
            )
            
            # Macro F1 (per paper Table 6)
            type_f1_macro = f1_score(
                all_type_labels[tracking_mask],
                all_type_preds[tracking_mask],
                labels=[1, 2, 3, 4],
                average='macro',
                zero_division=0
            )
        else:
            type_f1_per_class = np.zeros(4)
            type_f1_macro = 0.0
        
        return {
            "loss":  total_loss / n_batches,
            "det_precision": det_precision,
            "det_recall": det_recall,
            "det_f1": det_f1,
            "det_auc": det_auc,
            "type_f1_storage": type_f1_per_class[0] if len(type_f1_per_class) > 0 else 0,
            "type_f1_beacon": type_f1_per_class[1] if len(type_f1_per_class) > 1 else 0,
            "type_f1_fingerprint": type_f1_per_class[2] if len(type_f1_per_class) > 2 else 0,
            "type_f1_conversion": type_f1_per_class[3] if len(type_f1_per_class) > 3 else 0,
            "type_f1_macro":  type_f1_macro,
            "predictions": {
                "det_preds": all_det_preds,
                "det_probs": all_det_probs,
                "det_labels": all_det_labels,
                "type_preds": all_type_preds,
                "type_labels": all_type_labels
            }
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ) -> Dict[str, List[float]]: 
        """
        Complete training loop with early stopping.
        
        Returns training history. 
        """
        if num_epochs is None:
            num_epochs = self. config.num_epochs
        
        early_stopping = EarlyStopping(
            patience=self. config.patience,
            min_delta=self.config.min_delta,
            mode='max'
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            combined_f1 = 0.6 * val_metrics["det_f1"] + 0.4 * val_metrics["type_f1_macro"]
            self.scheduler.step(combined_f1)
            current_lr = self. optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss']. append(train_metrics['loss_total'])
            self.history['train_det_loss'].append(train_metrics['loss_det'])
            self.history['train_type_loss'].append(train_metrics['loss_type'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_det_f1'].append(val_metrics['det_f1'])
            self.history['val_type_f1_macro'].append(val_metrics['type_f1_macro'])
            self.history['learning_rate'].append(current_lr)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss_total']:.4f} | "
                f"Val Loss: {val_metrics['loss']:. 4f} | "
                f"Det F1: {val_metrics['det_f1']:.4f} | "
                f"Type F1: {val_metrics['type_f1_macro']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Early stopping check
            if early_stopping(combined_f1, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        early_stopping.load_best_model(self.model)
        logger.info(f"Loaded best model with score: {early_stopping.best_score:. 4f}")
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model. state_dict(),
            'optimizer_state_dict': self. optimizer.state_dict(),
            'config': self.config. to_dict(),
            'history': self. history
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint. get('history', self.history)
        logger.info(f"Model loaded from {path}")


def cross_validate(
    features: np.ndarray,
    tracking_labels: np.ndarray,
    subtype_labels: np.ndarray,
    config: TrainingConfig = None,
    n_folds: int = 5
) -> Dict[str, List[float]]: 
    """
    Perform 5-fold cross-validation as described in Section 4.1.
    
    Returns average metrics across folds. 
    """
    config = config or TrainingConfig()
    device = torch.device("cuda" if torch. cuda.is_available() else "cpu")
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)
    
    cv_results = {
        'det_precision': [], 'det_recall': [], 'det_f1': [],
        'type_f1_macro':  []
    }
    
    scaler = StandardScaler()
    
    for fold, (train_idx, val_idx) in enumerate(kfold. split(features, tracking_labels)):
        logger.info(f"Fold {fold+1}/{n_folds}")
        
        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        y_det_train, y_det_val = tracking_labels[train_idx], tracking_labels[val_idx]
        y_type_train, y_type_val = subtype_labels[train_idx], subtype_labels[val_idx]
        
        # Scale features
        X_train = scaler.fit_transform(X_train)
        X_val = scaler. transform(X_val)
        
        # Create datasets
        train_dataset = FunctionFeatureDataset(X_train, y_det_train, y_type_train)
        val_dataset = FunctionFeatureDataset(X_val, y_det_val, y_type_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        # Create model
        model = create_model(
            input_dim=X_train.shape[1],
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        )
        
        # Train
        trainer = MTLTrainer(model, config, device)
        trainer.train(train_loader, val_loader, num_epochs=50)  # Fewer epochs for CV
        
        # Evaluate
        metrics = trainer.evaluate(val_loader)
        
        cv_results['det_precision'].append(metrics['det_precision'])
        cv_results['det_recall']. append(metrics['det_recall'])
        cv_results['det_f1'].append(metrics['det_f1'])
        cv_results['type_f1_macro'].append(metrics['type_f1_macro'])
    
    # Compute averages and std
    logger.info("\n=== Cross-Validation Results ===")
    for metric, values in cv_results.items():
        mean_val = np. mean(values)
        std_val = np.std(values)
        logger.info(f"{metric}: {mean_val:.4f} (±{std_val:.4f})")
    
    return cv_results


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MTL-TABlock Training")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir. mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = Tra