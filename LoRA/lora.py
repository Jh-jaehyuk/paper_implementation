import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

from torch.utils.data import DataLoader


class LoRALayer(nn.Module):
    """
    LoRA 레이어 구현
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16,
            dropout_p: float = 0.1,
            merge_weights: bool = False
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights

        # LoRA specific layers
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        # Optional dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Weight initialization"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA path
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return lora_output


class LoRALinear(nn.Module):
    """
    LoRA가 적용된 Linear 레이어
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16,
            dropout_p: float = 0.1,
            merge_weights: bool = False
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout_p=dropout_p,
            merge_weights=merge_weights
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class LoRATransformerLayer(nn.Module):
    """
    LoRA가 적용된 Transformer 레이어
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            lora_rank: int = 8,
            lora_alpha: float = 16,
            lora_dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention with LoRA
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.q_proj = LoRALinear(d_model, d_model, lora_rank, lora_alpha, lora_dropout)
        self.k_proj = LoRALinear(d_model, d_model, lora_rank, lora_alpha, lora_dropout)
        self.v_proj = LoRALinear(d_model, d_model, lora_rank, lora_alpha, lora_dropout)

        # Feed-forward network with LoRA
        self.ff1 = LoRALinear(d_model, dim_feedforward, lora_rank, lora_alpha, lora_dropout)
        self.ff2 = LoRALinear(dim_feedforward, d_model, lora_rank, lora_alpha, lora_dropout)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        src2 = self.norm1(src)
        q = self.q_proj(src2)
        k = self.k_proj(src2)
        v = self.v_proj(src2)
        src2 = self.self_attn(q, k, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)

        # Feed-forward
        src2 = self.norm2(src)
        src2 = self.ff2(F.relu(self.ff1(src2)))
        src = src + self.dropout(src2)

        return src


class LoRAModel(nn.Module):
    """
    LoRA가 적용된 전체 모델
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            lora_rank: int = 8,
            lora_alpha: float = 16,
            lora_dropout: float = 0.1
    ):
        super().__init__()

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer layers with LoRA
        self.layers = nn.ModuleList([
            LoRATransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embedding and positional encoding
        x = self.embedding(src)
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)

        # Output projection
        x = self.output_layer(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Transformer의 Positional Encoding
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device: str = 'mps'
):
    """
    LoRA 모델 학습 함수
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')


# 사용 예시
def example_usage():
    # 모델 파라미터
    vocab_size = 32000
    d_model = 768
    nhead = 12
    num_layers = 6

    # LoRA 파라미터
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1

    # 디바이스 설정
    device = torch.device('mps')

    # 모델 초기화
    model = LoRAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    ).to(device)

    # 옵티마이저 및 손실 함수 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


model, optimizer, criterion = example_usage()
