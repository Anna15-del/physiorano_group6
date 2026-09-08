"""
PhysioRANO Longitudinal Temporal Sequence Model (Phase 4 Branch 4B).

Implements PyTorch temporal sequence modeling (LSTM / GRU network) to track
longitudinal tumor volumetric trajectories and time-series follow-up scan dynamics.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("PhysioRANO.TemporalModel")


class TumorTemporalSequenceModel(nn.Module):
    """PyTorch LSTM/GRU sequence model for tracking longitudinal scan trajectories."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 32,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim (int): Input feature size per scan timepoint (e.g. volumes, delta_t, radiomics).
            hidden_dim (int): Hidden dimension size of recurrent layers.
            num_layers (int): Number of stacked LSTM layers.
            output_dim (int): Dimension of extracted temporal trajectory embedding.
            dropout (float): Dropout regularization probability.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        logger.info(f"Initialized TumorTemporalSequenceModel (input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim})")

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass processing sequence of scan timepoints.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [batch_size, seq_len, input_dim].
            lengths (Optional[torch.Tensor]): Valid sequence lengths per batch item.

        Returns:
            torch.Tensor: Extracted temporal sequence embedding of shape [batch_size, output_dim].
        """
        # LSTM output: [batch_size, seq_len, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Extract final hidden state corresponding to last timepoint
        if lengths is not None:
            batch_size = x.size(0)
            last_indices = (lengths - 1).clamp(min=0).long()
            last_hidden = lstm_out[torch.arange(batch_size), last_indices, :]
        else:
            last_hidden = lstm_out[:, -1, :]

        embeddings = self.fc(last_hidden)
        return embeddings


def extract_patient_temporal_features(df_patient: Dict) -> torch.Tensor:
    """Utility converting patient clinical/scan row into a temporal sequence tensor.

    Args:
        df_patient (Dict): Single patient record dictionary.

    Returns:
        torch.Tensor: Formatted sequence tensor [1, max_seq_len, 5].
    """
    num_scans = int(df_patient.get("num_followup_scans", 1))
    num_scans = max(1, min(num_scans, 10))

    # Construct synthetic/simulated volumetric trajectory per timepoint scan
    seq_features = []
    base_wt = 25.0 + 5.0 * np.random.randn()
    base_tc = 10.0 + 2.0 * np.random.randn()
    base_et = 5.0 + 1.0 * np.random.randn()

    for t_idx in range(num_scans):
        delta_t_days = (t_idx + 1) * 30.0
        # Volume growth or regression simulation
        growth_factor = 1.0 + 0.05 * t_idx
        wt_vol = max(1.0, base_wt * growth_factor)
        tc_vol = max(0.5, base_tc * growth_factor)
        et_vol = max(0.1, base_et * growth_factor)

        seq_features.append([delta_t_days, wt_vol, tc_vol, et_vol, float(t_idx + 1)])

    tensor_seq = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0)
    return tensor_seq


if __name__ == "__main__":
    model = TumorTemporalSequenceModel(input_dim=5, hidden_dim=64, output_dim=32)
    sample_seq = torch.randn(4, 5, 5)  # batch 4, 5 timepoints, 5 features
    out = model(sample_seq)
    print(f"Temporal Sequence Model Output Shape: {out.shape}")
