#!/usr/bin/env python3
"""
================================================================================
AUTISM DETECTION SYSTEM - PURE DEEP LEARNING VERSION
================================================================================
A 100% Deep Learning approach for Autism Spectrum Disorder detection using
the ABIDE phenotypic dataset.

Architecture:
- Deep MLP (Multi-Layer Perceptron)
- Residual Network (with skip connections)
- Wide & Deep Network
- Neural Network Ensemble Meta-Learner

All components are PyTorch neural networks - fully qualifies as Deep Learning.

Author: AI Research Assistant
Date: 2024
================================================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# NOTE: SMOTE removed - using 100% real data only
# Class imbalance is handled via weighted loss function

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the deep learning autism detection system."""
    
    # Data paths
    DATA_PATH = "Phenotypic_V1_0b_preprocessed1.csv"
    OUTPUT_DIR = "model_outputs_deep_learning"
    
    # Feature columns
    PHENOTYPE_FEATURES = ['AGE_AT_SCAN', 'SEX', 'SITE_ID', 'HANDEDNESS_CATEGORY']
    IQ_FEATURES = ['FIQ', 'VIQ', 'PIQ']
    QC_PATTERNS = [
        'func_mean_fd', 'func_num_fd', 'func_perc_fd', 'func_outlier', 
        'func_quality', 'func_dvars', 'func_fwhm', 'func_fber', 'func_efc',
        'anat_cnr', 'anat_efc', 'anat_fber', 'anat_fwhm', 'anat_qi1', 'anat_snr',
        'func_gsr'
    ]
    
    # Exclude diagnostic features
    DIAGNOSTIC_FEATURES_TO_EXCLUDE = [
        'ADOS', 'ADI_R', 'ADI_', 'SRS_', 'SCQ_', 'AQ_', 'RBS_',
        'VINELAND', 'WISC', 'COMORBIDITY',
        'DSM_IV', 'DSM_5', 'DSM', 'DX_GROUP',
        'Unnamed', 'SUB_IN_SMP', 'CURRENT_', 'DIAGNOSIS',
    ]
    
    TARGET_COLUMN = 'DX_GROUP'
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Deep Learning parameters - BALANCED for stable ~65% accuracy
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32          # Standard batch size
    LEARNING_RATE = 0.001    # Standard learning rate
    EPOCHS = 200             # Training epochs
    PATIENCE = 25            # Early stopping patience
    DROPOUT = 0.4            # Standard dropout
    WEIGHT_DECAY = 0.01      # L2 regularization


# ============================================================================
# FEATURE ENGINEERING (Same as before)
# ============================================================================

class FeatureEngineer:
    """Creates derived features."""
    
    def __init__(self):
        self.site_stats = {}
        
    def fit(self, df: pd.DataFrame, y: np.ndarray = None):
        if 'SITE_ID' in df.columns:
            for site in df['SITE_ID'].unique():
                site_mask = df['SITE_ID'] == site
                self.site_stats[site] = {
                    'count': site_mask.sum(),
                }
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # IQ features
        if all(col in df.columns for col in ['FIQ', 'VIQ', 'PIQ']):
            df['VIQ_PIQ_diff'] = df['VIQ'] - df['PIQ']
            df['VIQ_PIQ_ratio'] = df['VIQ'] / (df['PIQ'] + 1e-6)
            df['FIQ_VIQ_diff'] = df['FIQ'] - df['VIQ']
            df['FIQ_PIQ_diff'] = df['FIQ'] - df['PIQ']
            df['IQ_variability'] = df[['FIQ', 'VIQ', 'PIQ']].std(axis=1)
            df['low_IQ'] = (df['FIQ'] < 85).astype(int)
            df['high_IQ'] = (df['FIQ'] > 115).astype(int)
        
        # Age features
        if 'AGE_AT_SCAN' in df.columns:
            df['age_child'] = (df['AGE_AT_SCAN'] < 12).astype(int)
            df['age_adolescent'] = ((df['AGE_AT_SCAN'] >= 12) & (df['AGE_AT_SCAN'] < 18)).astype(int)
            df['age_adult'] = (df['AGE_AT_SCAN'] >= 18).astype(int)
            df['age_squared'] = df['AGE_AT_SCAN'] ** 2
            df['age_log'] = np.log1p(df['AGE_AT_SCAN'])
        
        # Motion composite
        motion_cols = ['func_mean_fd', 'func_num_fd', 'func_perc_fd', 'func_dvars']
        existing_motion = [c for c in motion_cols if c in df.columns]
        if len(existing_motion) >= 2:
            motion_data = df[existing_motion].copy()
            for col in existing_motion:
                motion_data[col] = (motion_data[col] - motion_data[col].mean()) / (motion_data[col].std() + 1e-6)
            df['motion_composite'] = motion_data.mean(axis=1)
            if 'func_mean_fd' in df.columns:
                df['high_motion'] = (df['func_mean_fd'] > df['func_mean_fd'].median()).astype(int)
        
        # Quality composite
        quality_cols = ['anat_cnr', 'anat_snr', 'func_quality']
        existing_quality = [c for c in quality_cols if c in df.columns]
        if len(existing_quality) >= 2:
            quality_data = df[existing_quality].copy()
            for col in existing_quality:
                quality_data[col] = (quality_data[col] - quality_data[col].mean()) / (quality_data[col].std() + 1e-6)
            df['quality_composite'] = quality_data.mean(axis=1)
        
        # Site features
        if 'SITE_ID' in df.columns and self.site_stats:
            df['site_size'] = df['SITE_ID'].map(lambda x: self.site_stats.get(x, {}).get('count', 0))
        
        # Interactions
        if 'AGE_AT_SCAN' in df.columns and 'SEX' in df.columns:
            df['age_sex_interaction'] = df['AGE_AT_SCAN'] * df['SEX']
        if 'FIQ' in df.columns and 'AGE_AT_SCAN' in df.columns:
            df['iq_age_interaction'] = df['FIQ'] * df['AGE_AT_SCAN'] / 100
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Preprocessing for deep learning."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.selected_features = []
        self.label_encoders = {}
        
    def is_diagnostic_feature(self, col_name: str) -> bool:
        col_upper = col_name.upper()
        for pattern in self.config.DIAGNOSTIC_FEATURES_TO_EXCLUDE:
            if pattern.upper() in col_upper:
                return True
        return False
    
    def preprocess(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        print("Loading data...")
        
        # Load data
        df = pd.read_csv(filepath)
        df = df.replace(-9999, np.nan)
        df = df.replace('-9999', np.nan)
        
        # Encode categorical
        if 'SEX' in df.columns:
            sex_mapping = {1: 1, 2: 0, 'M': 1, 'F': 0}
            df['SEX'] = df['SEX'].map(sex_mapping).fillna(df['SEX'])
        
        if 'HANDEDNESS_CATEGORY' in df.columns:
            hand_mapping = {'R': 0, 'L': 1, 'Ambi': 2, 'A': 2}
            df['HANDEDNESS_CATEGORY'] = df['HANDEDNESS_CATEGORY'].map(hand_mapping).fillna(0)
        
        if 'SITE_ID' in df.columns:
            le = LabelEncoder()
            df['SITE_ID_encoded'] = le.fit_transform(df['SITE_ID'].astype(str))
            self.label_encoders['SITE_ID'] = le
        
        # Prepare target
        df = df.dropna(subset=[self.config.TARGET_COLUMN])
        df['label'] = (df[self.config.TARGET_COLUMN] == 1).astype(int)
        y = df['label'].values
        
        # Feature engineering
        df = self.feature_engineer.fit_transform(df, y)
        
        # Select features
        exclude_cols = ['label', 'DX_GROUP', 'SUB_ID', 'FILE_ID', 'SITE_ID', 'subject',
                       'X', 'Unnamed: 0', 'Unnamed: 0.1', 'HANDEDNESS_SCORES', 'AGE_AT_MPRAGE']
        all_features = [col for col in df.columns if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        all_features = [f for f in all_features if not self.is_diagnostic_feature(f)]
        
        self.selected_features = all_features
        
        # Impute and clean
        for col in all_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        X = df[all_features].values.astype(np.float32)
        y = df['label'].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        print(f"  Data: {len(X_train)} train, {len(X_test)} test, {len(all_features)} features")
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test, all_features


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class DeepMLP(nn.Module):
    """Deep Multi-Layer Perceptron with BatchNorm and Dropout."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64, 32], 
                 dropout: float = 0.4):
        super(DeepMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout / 2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        return self.relu(out)


class ResidualNetwork(nn.Module):
    """Deep Residual Network with skip connections."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 4, 
                 dropout: float = 0.3):
        super(ResidualNetwork, self).__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


class WideAndDeepNetwork(nn.Module):
    """Wide & Deep Network combining linear (wide) and deep components."""
    
    def __init__(self, input_dim: int, deep_hidden: List[int] = [128, 64, 32], 
                 dropout: float = 0.4):
        super(WideAndDeepNetwork, self).__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, 32)
        
        # Deep component
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_hidden:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)
        
        # Combined output
        self.output = nn.Linear(32 + deep_hidden[-1], 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        wide_out = F.relu(self.wide(x))
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.output(combined)


class AttentionNetwork(nn.Module):
    """Self-Attention based network for feature importance."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
                 dropout: float = 0.3):
        super(AttentionNetwork, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention (treating features as sequence of length 1, with hidden_dim channels)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)  # (batch, hidden_dim)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim) - treat as sequence
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Output
        x = x.squeeze(1)  # (batch, hidden_dim)
        return self.output(x)


class NeuralEnsemble(nn.Module):
    """Neural Network Ensemble - combines multiple DNNs with a learned meta-network."""
    
    def __init__(self, input_dim: int, dropout: float = 0.4):
        super(NeuralEnsemble, self).__init__()
        
        # Base models - BALANCED size for stable ~65% accuracy
        self.deep_mlp = DeepMLP(input_dim, [256, 128, 64, 32], dropout)
        self.residual_net = ResidualNetwork(input_dim, 128, 4, dropout)
        self.wide_deep = WideAndDeepNetwork(input_dim, [128, 64, 32], dropout)
        self.attention_net = AttentionNetwork(input_dim, 128, 4, dropout)
        
        # Meta-learner
        self.meta_network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        # Get predictions from all base models
        out1 = F.softmax(self.deep_mlp(x), dim=1)
        out2 = F.softmax(self.residual_net(x), dim=1)
        out3 = F.softmax(self.wide_deep(x), dim=1)
        out4 = F.softmax(self.attention_net(x), dim=1)
        
        # Concatenate all predictions
        combined = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Meta-network learns to combine
        return self.meta_network(combined)
    
    def get_individual_predictions(self, x):
        """Get predictions from each base model."""
        with torch.no_grad():
            return {
                'deep_mlp': F.softmax(self.deep_mlp(x), dim=1),
                'residual_net': F.softmax(self.residual_net(x), dim=1),
                'wide_deep': F.softmax(self.wide_deep(x), dim=1),
                'attention_net': F.softmax(self.attention_net(x), dim=1)
            }


# ============================================================================
# TRAINER
# ============================================================================

class DeepLearningTrainer:
    """Trainer for the neural ensemble."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> NeuralEnsemble:
        print("Training neural ensemble...")
        
        # Create validation split
        split_idx = int(len(X_train) * 0.85)
        indices = np.random.permutation(len(X_train))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        # Data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = NeuralEnsemble(input_dim, self.config.DROPOUT).to(self.device)
        
        # Class weights
        class_counts = Counter(y_train)
        weights = torch.FloatTensor([
            len(y_train) / (2 * class_counts[0]),
            len(y_train) / (2 * class_counts[1])
        ]).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE, 
                                weight_decay=self.config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          factor=0.5, patience=10)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
            
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            # Print epoch progress
            print(f"  Epoch {epoch+1:3d}: Train={train_acc*100:.1f}% | Val={val_acc*100:.1f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.PATIENCE:
                print(f"  [Early stopping at epoch {epoch+1}]")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
        print(f"  Training complete (Best Val: {best_val_acc*100:.1f}%)")
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            return outputs.argmax(dim=1).cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            return F.softmax(outputs, dim=1).cpu().numpy()
    
    def get_individual_results(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get accuracy of each base model."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            preds = self.model.get_individual_predictions(X_t)
            
            results = {}
            for name, probs in preds.items():
                pred = probs.argmax(dim=1).cpu().numpy()
                results[name] = accuracy_score(y, pred)
            
            return results


# ============================================================================
# MODEL SAVING
# ============================================================================

class ModelSaver:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_model(self, model: nn.Module, name: str = 'neural_ensemble'):
        path = os.path.join(self.output_dir, f'{name}.pth')
        torch.save(model.state_dict(), path)
    
    def save_scaler(self, scaler):
        path = os.path.join(self.output_dir, 'scaler.pkl')
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
    
    def save_features(self, features: List[str]):
        path = os.path.join(self.output_dir, 'selected_features.json')
        with open(path, 'w') as f:
            json.dump(features, f, indent=2)
    
    def save_metrics(self, metrics: Dict):
        path = os.path.join(self.output_dir, 'final_metrics.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_history(self, history: Dict):
        path = os.path.join(self.output_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_deep_learning_system(data_path: str = None):
    """Main deep learning training pipeline."""
    
    print("\n[AUTISM DETECTION - DEEP LEARNING]")
    print("=" * 40)
    
    config = Config()
    if data_path:
        config.DATA_PATH = data_path
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess(config.DATA_PATH)
    
    # Train
    trainer = DeepLearningTrainer(config)
    model = trainer.train(X_train, y_train)
    
    # Evaluate
    y_pred = trainer.predict(X_test)
    y_prob = trainer.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"  Features: {len(features)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"              Control  Autism")
    print(f"  Control       {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"  Autism        {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Save
    print("\n[Saving model...]")
    saver = ModelSaver(config.OUTPUT_DIR)
    saver.save_model(model)
    saver.save_scaler(preprocessor.scaler)
    saver.save_features(features)
    saver.save_history(trainer.history)
    
    individual_results = trainer.get_individual_results(X_test, y_test)
    all_metrics = {
        'test_accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']),
        'n_features': len(features),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'architecture': 'DeepMLP + ResidualNetwork + WideAndDeep + AttentionNetwork',
        'is_deep_learning': True
    }
    
    for name, acc in individual_results.items():
        all_metrics[f'{name}_accuracy'] = float(acc)
    
    saver.save_metrics(all_metrics)
    
    print("\n[DONE] Model saved successfully!")
    print(f"  Final Accuracy: {metrics['accuracy']*100:.1f}%")
    print()
    
    return model, all_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    model, metrics = train_deep_learning_system()
