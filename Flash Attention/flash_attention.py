import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size=64 ):
        super(FlashAttention, self).__init__()
        self.embed_dim = embed_dim # 입력 임베딩 차원
        self.num_heads = num_heads # 멀티 헤드 수
        self.block_size = block_size # 블록 크기 (Attention을 블록 단위로 처리)

        # Projection layers for query, key, and value
        # Query, Key, Value에 대한 선형 변환 레이어
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection layer
        # Attention의 최종 출력을 변환하는 선형 레이어
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.size()

        # Linear projections for Q, K, and V
        # Query, Key, Value 로 입력 데이터 변환
        # 결과는 [batch_size, seq_len, num_heads, head_dim]으로 재구성
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Split into blocks to reduce memory usage
        # 블록 단위로 처리된 Attention 출력을 저장할 리스트
        output = []

        # 입력 시퀀스를 블록 단위로 나눠서 처리
        for i in range(0, seq_len, self.block_size):
            # 현재 블록의 Query, Key, Value를 슬라이싱
            q_block = q[:, :, i:i + self.block_size, :] # [batch_size, num_heads, block_size, head_dim]
            k_block = k[:, :, i:i + self.block_size, :] # [batch_size, num_heads, block_size, head_dim]
            v_block = v[:, :, i:i + self.block_size, :] # [batch_size, num_heads, block_size, head_dim]

            # Scaled dot-product attention
            # Query와 Key의 내적 계산 후, 크기 스케일링
            attn_weights = torch.matmul(q_block, k_block.transpose(-2, -1)) / (q_block.size(-1) ** 0.5)

            # Softmax로 Attention 가중치 계산
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Apply attention weights to values
            # Attention 가중치를 Value에 적용하여 출력 계산
            attn_output = torch.matmul(attn_weights, v_block)

            # 현재 블록의 출력을 리스트에 추가
            output.append(attn_output)

        # Concatenate block outputs and project to final dimensions
        # 모든 블록의 출력을 결합 (원래 순서로 복원)
        output = torch.cat(output, dim=2).transpose(1, 2).contiguous()

        # 병합된 출력 데이터를 원래 임베딩 차원으로 변환
        output = output.view(batch_size, seq_len, -1)

        # 최종 출력 반환
        return self.out_proj(output)


# Example usage
batch_size = 8 # 배치 크기
seq_len = 128 # 시퀀스 길이
embed_dim = 64 # 임베딩 차원
num_heads = 8 # 멀티 헤드 수

# 임의로 입력 데이터 생성
x = torch.rand(batch_size, seq_len, embed_dim)

# Flash Attention 모델 생성
flash_attention = FlashAttention(embed_dim, num_heads)

# 모델 실행
output = flash_attention(x)

# 입력 및 출력의 크기 확인
print("Input shape:", x.shape) # [batch_size, seq_len, embed_dim]
print("Output shape:", output.shape) # [batch_size, seq_len, embed_dim]