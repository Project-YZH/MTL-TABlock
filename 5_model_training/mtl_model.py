"""
MTL-TABlock:  Multi-Task Learning Model

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class SubtypeClass(IntEnum):
    """Tracking function subtype classes."""
    BENIGN = 0
    STORAGE_TRACKING = 1
    NETWORK_BEACON = 2
    FINGERPRINTING = 3
    CONVERSION_ANALYTICS = 4


# Subtype mapping constants
SUBTYPE_LABELS = {
    0: "benign",
    1: "storage_tracking",
    2: "network_beacon",
    3: "fingerprinting",
    4: "conversion_analytics"
}

SUBTYPE_TO_IDX = {v:  k for k, v in SUBTYPE_LABELS.items()}

IDX_TO_SUBTYPE = SUBTYPE_LABELS


@dataclass
class ModelConfig:
    """Configuration for MTL-TABlock model."""
    input_dim: int
    hidden_dims: List[int] = None
    num_subtypes: int = 5
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    activation: str = "relu"
    
    def __post_init__(self):
        if self.hidden_dims is None: 
            self.hidden_dims = [256, 128, 64]


class SharedRepresentationLayer(nn.Module):
    """
    Shared representation layer f_θ(·) as described in Section 3.5.1.
    
    Maps input features to a low-dimensional semantic space using
    a multi-layer perceptron structure with L hidden layers.
    
    Architecture per layer:
        h_i^(l) = φ(W^(l) · h_i^(l-1) + b^(l))
    
    where φ(·) is ReLU activation function. 
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        """
        Initialize the shared representation layer.
        
        Args: 
            input_dim:  Dimension of input features (d)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability (default 0.3 per paper)
            use_batch_norm: Whether to use batch normalization
            activation:  Activation function ("relu", "leaky_relu", "gelu")
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self. hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn. Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional, helps with training stability)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == "relu":
                layers.append(nn. ReLU(inplace=True))
            elif activation == "leaky_relu":
                layers. append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == "gelu": 
                layers.append(nn.GELU())
            else:
                layers.append(nn. ReLU(inplace=True))
            
            # Dropout for regularization
            layers.append(nn. Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.network = nn. Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn. Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared layers.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns: 
            Shared representation h_i [batch_size, output_dim]
        """
        return self.network(x)


class TrackingDetectionHead(nn.Module):
    """
    Primary task head for tracking function detection. 
    
    Binary classification:  tracking (1) vs non-tracking (0)
    
    Formula from Section 3.5.1:
        p_det = σ(w_det · h + b_det)
    
    where σ(·) is the sigmoid function.
    """
    
    def __init__(self, input_dim:  int, hidden_dim: int = None):
        """
        Initialize detection head. 
        
        Args:
            input_dim:  Dimension of shared representation
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()
        
        if hidden_dim is not None:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn. Linear(hidden_dim, 1)
            )
        else:
            self. network = nn.Linear(input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn. Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, h:  torch.Tensor) -> torch.Tensor:
        """
        Predict tracking probability.
        
        Args:
            h: Shared representation [batch_size, hidden_dim]
            
        Returns: 
            Tracking probability p_det [batch_size, 1]
        """
        logits = self.network(h)
        return torch.sigmoid(logits)
    
    def forward_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return raw logits without sigmoid."""
        return self.network(h)


class SubtypeIdentificationHead(nn.Module):
    """
    Auxiliary task head for tracking function subtype identification. 
    
    5-class classification: benign + 4 tracking subtypes
    
    Formula from Section 3.5.1:
        p_type = softmax(W_type · h + b_type)
    
    Classes:
        0: Benign (non-tracking)
        1: Storage Tracking
        2: Network Beacon
        3: Fingerprinting
        4: Conversion Analytics
    """
    
    def __init__(self, input_dim: int, num_classes: int = 5, hidden_dim: int = None):
        """
        Initialize subtype identification head.
        
        Args: 
            input_dim:  Dimension of shared representation
            num_classes: Number of subtype classes (default 5)
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        if hidden_dim is not None:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn. Dropout(0.2),
                nn. Linear(hidden_dim, num_classes)
            )
        else:
            self.network = nn. Linear(input_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self. modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module. weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict subtype probabilities.
        
        Args:
            h: Shared representation [batch_size, hidden_dim]
            
        Returns: 
            Subtype probabilities p_type^k [batch_size, num_classes]
        """
        logits = self.network(h)
        return F.softmax(logits, dim=-1)
    
    def forward_logits(self, h:  torch.Tensor) -> torch.Tensor:
        """Return raw logits without softmax."""
        return self.network(h)


class MTLTABlockModel(nn.Module):
    """
    MTL-TABlock:  Multi-Task Learning Model for Type-Aware Tracking Function Blocking.
    
    This model implements the multi-task learning framework described in Section 3.5.
    
    Architecture:
        Input features x_i ∈ R^d
            ↓
        Shared Representation Layer f_θ(·)
            ↓
        Shared representation h_i
            ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
    Detection Head              Subtype Head
    (Binary classification)     (5-class classification)
        ↓                           ↓
    p_det (tracking prob)       p_type (subtype probs)
    
    The model is trained jointly with weighted loss: 
        L_total = λ_det * L_det + λ_type * L_type
    """
    
    def __init__(
        self,
        input_dim:  int,
        hidden_dims: List[int] = None,
        num_subtypes: int = 5,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        head_hidden_dim: int = None,
        config: ModelConfig = None
    ):
        """
        Initialize MTL-TABlock model.
        
        Args:
            input_dim: Number of input features (structural + contextual)
            hidden_dims: List of hidden dimensions for shared layer
            num_subtypes: Number of subtype classes (5:  benign + 4 tracking types)
            dropout_rate: Dropout rate (0.3 per paper Section 3.5.2)
            use_batch_norm:  Whether to use batch normalization
            head_hidden_dim: Optional hidden dimension for task heads
            config:  Optional ModelConfig object
        """
        super().__init__()
        
        if config is not None:
            input_dim = config. input_dim
            hidden_dims = config.hidden_dims
            num_subtypes = config.num_subtypes
            dropout_rate = config. dropout_rate
            use_batch_norm = config.use_batch_norm
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_subtypes = num_subtypes
        
        # Shared representation layer
        self.shared_layer = SharedRepresentationLayer(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Task-specific heads (independent parameters per paper)
        self.detection_head = TrackingDetectionHead(
            input_dim=hidden_dims[-1],
            hidden_dim=head_hidden_dim
        )
        self.subtype_head = SubtypeIdentificationHead(
            input_dim=hidden_dims[-1],
            num_classes=num_subtypes,
            hidden_dim=head_hidden_dim
        )
    
    def forward(
        self,
        x: torch. Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Forward pass through MTL model.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Tuple of: 
                - p_det: Detection probabilities [batch_size, 1]
                - p_type:  Subtype probabilities [batch_size, num_subtypes]
        """
        # Shared representation
        h = self.shared_layer(x)
        
        # Task-specific outputs
        p_det = self. detection_head(h)
        p_type = self.subtype_head(h)
        
        return p_det, p_type
    
    def forward_with_logits(
        self,
        x: torch. Tensor
    ) -> Tuple[torch. Tensor, torch. Tensor, torch.Tensor, torch.Tensor]: 
        """
        Forward pass returning both probabilities and logits. 
        
        Args:
            x:  Input features [batch_size, input_dim]
            
        Returns:
            Tuple of (p_det, p_type, logits_det, logits_type)
        """
        h = self.shared_layer(x)
        
        logits_det = self.detection_head. forward_logits(h)
        logits_type = self.subtype_head.forward_logits(h)
        
        p_det = torch.sigmoid(logits_det)
        p_type = F.softmax(logits_type, dim=-1)
        
        return p_det, p_type, logits_det, logits_type
    
    def predict(
        self,
        x: torch.Tensor,
        detection_threshold: float = 0.5
    ) -> Tuple[torch. Tensor, torch. Tensor, torch.Tensor, torch.Tensor]: 
        """
        Make predictions for deployment.
        
        According to Section 3.5.1: 
        - First determine if function is tracking based on p_det >= τ
        - Only if tracking, select subtype with highest probability
        
        Args:
            x: Input features [batch_size, input_dim]
            detection_threshold:  Threshold τ for tracking detection
            
        Returns:
            Tuple of: 
                - is_tracking: Boolean tensor [batch_size]
                - subtypes: Predicted subtype indices [batch_size]
                - det_probs: Detection probabilities [batch_size]
                - type_probs:  Subtype probabilities [batch_size, num_subtypes]
        """
        self.eval()
        with torch.no_grad():
            p_det, p_type = self.forward(x)
            
            det_probs = p_det.squeeze(-1)
            is_tracking = det_probs >= detection_threshold
            
            # Get predicted subtypes
            subtypes = torch.argmax(p_type, dim=-1)
            
            # Only assign subtype if detected as tracking
            # Non-tracking functions get subtype 0 (benign)
            subtypes = torch.where(is_tracking, subtypes, torch.zeros_like(subtypes))
            
            return is_tracking, subtypes, det_probs, p_type
    
    def get_shared_representation(self, x: torch. Tensor) -> torch.Tensor:
        """
        Get the shared representation for input features.
        
        Useful for feature analysis and visualization.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Shared representation h [batch_size, hidden_dim]
        """
        return self.shared_layer(x)
    
    def count_parameters(self) -> Dict[str, int]: 
        """Count trainable parameters in each component."""
        shared_params = sum(p.numel() for p in self. shared_layer.parameters() if p.requires_grad)
        det_params = sum(p.numel() for p in self. detection_head.parameters() if p.requires_grad)
        type_params = sum(p.numel() for p in self. subtype_head. parameters() if p.requires_grad)
        
        return {
            "shared_layer": shared_params,
            "detection_head":  det_params,
            "subtype_head": type_params,
            "total":  shared_params + det_params + type_params
        }


class MTLLoss(nn.Module):
    """
    Combined loss function for MTL-TABlock. 
    
    From Section 3.5.1:
        L_total = λ_det * L_det + λ_type * L_type
    
    where:
        L_det = -Σ[y_i * log(p_det) + (1-y_i) * log(1-p_det)]  (Binary CE)
        L_type = -Σ Σ 1[t_i=k] * log(p_type^k)                  (Multinomial CE)
    
    Default weights from paper:
        λ_det = 1.0 (prioritize detection task)
        λ_type = 0.5 (auxiliary task)
    """
    
    def __init__(
        self,
        lambda_det: float = 1.0,
        lambda_type: float = 0.5,
        det_class_weights: Optional[torch.Tensor] = None,
        type_class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        use_focal_loss:  bool = False,
        focal_gamma: float = 2.0
    ):
        """
        Initialize MTL loss function.
        
        Args:
            lambda_det: Weight for detection loss (default 1.0)
            lambda_type: Weight for subtype loss (default 0.5)
            det_class_weights: Optional weights for detection classes (handle imbalance)
            type_class_weights: Optional weights for subtype classes
            label_smoothing: Label smoothing factor
            use_focal_loss: Whether to use focal loss for detection
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.lambda_det = lambda_det
        self.lambda_type = lambda_type
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # Detection loss (binary cross-entropy)
        if det_class_weights is not None:
            self.det_pos_weight = det_class_weights[1] / det_class_weights[0]
        else:
            self.det_pos_weight = None
        
        # Subtype loss (cross-entropy with optional class weights)
        self.type_loss = nn.CrossEntropyLoss(
            weight=type_class_weights,
            label_smoothing=label_smoothing
        )
    
    def focal_loss(
        self,
        pred: torch.Tensor,
        target:  torch.Tensor,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance in detection. 
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * bce).mean()
    
    def forward(
        self,
        p_det: torch.Tensor,
        p_type: torch. Tensor,
        y_det: torch. Tensor,
        y_type: torch.Tensor,
        mask_type: Optional[torch. Tensor] = None
    ) -> Tuple[torch. Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            p_det:  Predicted detection probabilities [batch_size, 1]
            p_type: Predicted subtype probabilities [batch_size, num_classes]
            y_det: Ground truth tracking labels (0 or 1) [batch_size]
            y_type: Ground truth subtype labels (0-4) [batch_size]
            mask_type: Optional mask for subtype loss (exclude unknown samples)
            
        Returns:
            Tuple of: 
                - Total loss tensor
                - Dictionary with individual loss values
        """
        # Detection loss
        p_det_squeezed = p_det.squeeze(-1)
        
        if self.use_focal_loss: 
            l_det = self. focal_loss(p_det_squeezed, y_det. float(), self.focal_gamma)
        else:
            if self.det_pos_weight is not None:
                l_det = F.binary_cross_entropy(
                    p_det_squeezed, 
                    y_det.float(),
                    reduction='mean'
                )
            else: 
                l_det = F.binary_cross_entropy(p_det_squeezed, y_det.float())
        
        # Subtype loss
        # Get logits from probabilities for cross-entropy
        # Note: We use log of softmax probabilities
        logits_type = torch.log(p_type + 1e-8)
        
        if mask_type is not None:
            # Only compute loss for samples with valid subtype labels
            if mask_type.sum() > 0:
                l_type = F.nll_loss(logits_type[mask_type], y_type[mask_type]. long())
            else: 
                l_type = torch.tensor(0.0, device=p_det.device)
        else: 
            l_type = F.nll_loss(logits_type, y_type.long())
        
        # Combined loss
        total_loss = self. lambda_det * l_det + self. lambda_type * l_type
        
        return total_loss, {
            "loss_det": l_det.item(),
            "loss_type":  l_type.item() if isinstance(l_type, torch.Tensor) else l_type,
            "loss_total": total_loss.item()
        }


class MTLLossWithLogits(nn.Module):
    """
    MTL Loss that works directly with logits (more numerically stable).
    """
    
    def __init__(
        self,
        lambda_det: float = 1.0,
        lambda_type: float = 0.5,
        det_pos_weight: Optional[torch.Tensor] = None,
        type_class_weights: Optional[torch.Tensor] = None,
        label_smoothing:  float = 0.0
    ):
        super().__init__()
        
        self.lambda_det = lambda_det
        self.lambda_type = lambda_type
        
        # Binary cross-entropy with logits
        self.det_loss = nn. BCEWithLogitsLoss(pos_weight=det_pos_weight)
        
        # Cross-entropy loss for subtype
        self.type_loss = nn.CrossEntropyLoss(
            weight=type_class_weights,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits_det: torch.Tensor,
        logits_type: torch.Tensor,
        y_det: torch. Tensor,
        y_type: torch. Tensor
    ) -> Tuple[torch. Tensor, Dict[str, float]]:
        """
        Compute loss from logits.
        
        Args: 
            logits_det: Detection logits [batch_size, 1]
            logits_type:  Subtype logits [batch_size, num_classes]
            y_det: Ground truth tracking labels [batch_size]
            y_type:  Ground truth subtype labels [batch_size]
            
        Returns:
            Total loss and loss dictionary
        """
        l_det = self. det_loss(logits_det. squeeze(-1), y_det.float())
        l_type = self.type_loss(logits_type, y_type.long())
        
        total_loss = self.lambda_det * l_det + self.lambda_type * l_type
        
        return total_loss, {
            "loss_det":  l_det.item(),
            "loss_type": l_type.item(),
            "loss_total": total_loss.item()
        }


def create_model(
    input_dim: int,
    hidden_dims:  List[int] = None,
    num_subtypes: int = 5,
    dropout_rate:  float = 0.3,
    **kwargs
) -> MTLTABlockModel: 
    """
    Factory function to create MTL-TABlock model.
    
    Args: 
        input_dim:  Number of input features
        hidden_dims: Hidden layer dimensions
        num_subtypes: Number of subtype classes
        dropout_rate:  Dropout rate
        **kwargs: Additional arguments for model
        
    Returns: 
        Initialized MTLTABlockModel
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    return MTLTABlockModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_subtypes=num_subtypes,
        dropout_rate=dropout_rate,
        **kwargs
    )


def load_model(
    checkpoint_path: str,
    input_dim: int,
    hidden_dims: List[int] = None,
    device: torch.device = None
) -> MTLTABlockModel: 
    """
    Load model from checkpoint. 
    
    Args: 
        checkpoint_path: Path to saved model
        input_dim:  Input feature dimension
        hidden_dims: Hidden layer dimensions
        device: Device to load model to
        
    Returns:
        Loaded MTLTABlockModel
    """
    if device is None: 
        device = torch. device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model