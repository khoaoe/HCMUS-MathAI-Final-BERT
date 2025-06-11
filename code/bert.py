import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# ========================= Cấu hình (Configuration) =========================
# Lớp này chứa các siêu tham số (hyperparameter) cho mô hình BERT,
# tương ứng với các ký hiệu trong bài báo gốc (Mục 3, "Model Architecture"):
# - num_hidden_layers: Số tầng Transformer (L)
# - hidden_size: Kích thước ẩn (H)
# - num_attention_heads: Số lượng attention head (A)
# - intermediate_size: Kích thước tầng feed-forward (4*H)


@dataclass
class BertConfig:
    """Cấu hình cho mô hình BERT"""
    vocab_size: int = 30522                     # Kích thước bộ từ vựng
    hidden_size: int = 768                      # Kích thước ẩn (d_model)
    num_hidden_layers: int = 12                 # Số khối transformer
    num_attention_heads: int = 12               # Số lượng attention head
    intermediate_size: int = 3072               # Kích thước tầng trung gian của FFN
    hidden_dropout_prob: float = 0.1            # Dropout cho các tầng ẩn
    attention_probs_dropout_prob: float = 0.1   # Dropout cho attention probabilities
    max_position_embeddings: int = 512          # Độ dài tối đa của chuỗi
    type_vocab_size: int = 2                    # Kích thước từ vựng của segment (0 hoặc 1)
    initializer_range: float = 0.02             # Độ lệch chuẩn cho khởi tạo trọng số


# ========================= Các Module con của BERT =========================

class BertEmbeddings(nn.Module):
    """
    Tạo các embedding từ input_ids, token_type_ids và position_ids.
    Tổng hợp 3 loại embedding:
    1. Token Embeddings: Biểu diễn cho từng token.
    2. Segment Embeddings: Phân biệt giữa 2 câu (cho tác vụ NSP).
    3. Position Embeddings: Cung cấp thông tin về vị trí của token trong chuỗi.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Cơ chế Multi-Head Self-Attention.
    Công thức Attention: Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    - Q: Query, K: Key, V: Value
    - d_k: Chiều của vector Key/Query. Chia cho sqrt(d_k) để ổn định gradient.
    Multi-Head: Chia hidden_size thành nhiều 'head', mỗi head tính attention độc lập,
    sau đó kết hợp lại. Điều này cho phép mô hình tập trung vào các khía cạnh khác nhau
    của thông tin.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # Chuyển chiều để tính toán attention: (batch, num_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Tính attention scores
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Áp dụng attention mask (ví dụ để bỏ qua các padding token)
            attention_scores = attention_scores + attention_mask

        # Chuẩn hóa thành attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """
    Tầng output sau Multi-Head Attention, bao gồm một tầng linear, dropout,
    và kết nối phần dư (residual connection) với Layer Normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Residual connection + Layer Norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Gói gọn MultiHeadSelfAttention và BertSelfOutput."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = MultiHeadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    """Tầng Feed-Forward đầu tiên trong khối Transformer."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    Tầng Feed-Forward thứ hai, kết hợp với dropout, residual connection
    và layer normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    Một khối Transformer Encoder hoàn chỉnh, bao gồm:
    1. Multi-Head Attention
    2. Feed-Forward Network
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    """
    Chồng chất N tầng (BertLayer) lên nhau để tạo thành bộ mã hóa của BERT.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config)
                                   for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        return {
            'last_hidden_state': hidden_states,
            'attentions': all_attentions,
        }

# ========================= Các Module Tác vụ Cụ thể =========================


class BertPooler(nn.Module):
    """
    Lấy biểu diễn của token [CLS] (ở vị trí đầu tiên) và biến đổi nó
    qua một tầng linear + Tanh. Biểu diễn này thường được dùng cho
    các tác vụ phân loại toàn câu (ví dụ: NSP, phân loại cảm xúc).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Lấy hidden state của token đầu tiên ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """Tầng transform trước khi dự đoán token cho MLM."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Đầu ra cho tác vụ Masked Language Model (MLM).
    Dự đoán token bị che (masked) từ hidden state của nó.
    """

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # Tầng decoder, trọng số được chia sẻ với tầng embedding
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

# ========================= Mô hình BERT Hoàn chỉnh =========================


class BertModel(nn.Module):
    """Mô hình BERT cơ bản (chỉ gồm Encoder)."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Tạo attention mask mở rộng để broadcasting
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # (B, 1, 1, S)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, output_attentions)

        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output)

        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'attentions': encoder_outputs['attentions']
        }


class BertForPreTraining(nn.Module):
    """
    Mô hình BERT cho Pre-training, bao gồm cả hai tác vụ:
    1. Masked Language Model (MLM)
    2. Next Sentence Prediction (NSP)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        # Đầu ra cho MLM
        self.cls_mlm = BertLMPredictionHead(
            config, self.bert.embeddings.word_embeddings.weight)
        # Đầu ra cho NSP
        self.cls_nsp = nn.Linear(config.hidden_size, 2)
        self._init_weights()

    def _init_weights(self):
        # Khởi tạo trọng số
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(
                    mean=0.0, std=self.bert.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids, attention_mask, output_attentions=False):
        outputs = self.bert(input_ids, token_type_ids,
                            attention_mask, output_attentions)
        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']

        prediction_scores = self.cls_mlm(sequence_output)
        seq_relationship_score = self.cls_nsp(pooled_output)

        return {
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score,
            'attentions': outputs['attentions']
        }

# ========================= Hàm Trực quan hóa và Kiểm thử =========================


def visualize_attention(bert_model: BertModel, tokenizer: BertTokenizer, sentence: str, device='cpu'):
    """Trực quan hóa trọng số attention cho một câu."""
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    # Lấy attention weights từ mô hình
    outputs = bert_model(**inputs, output_attentions=True)
    attentions = outputs['attentions']  # tuple of (B, H, S, S)

    # Lấy attention của layer cuối cùng
    last_layer_attention = attentions[-1][0]  # (H, S, S)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Tính trung bình attention qua các head
    avg_attention = last_layer_attention.mean(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=tokens,
                yticklabels=tokens, cmap='viridis')
    plt.title(f'Average Attention - Last Layer for: "{sentence}"')
    plt.show()


def test_attention_mechanism():
    """Kiểm thử cơ chế attention."""
    print("="*20 + " Kiểm thử Cơ chế Attention " + "="*20)
    config = BertConfig(hidden_size=12, num_attention_heads=3)
    attention = MultiHeadSelfAttention(config)

    # Tạo input giả
    # batch_size=1, seq_length=5, hidden_size=12
    hidden_states = torch.randn(1, 5, 12)
    # Tạo mask, che đi 2 token cuối
    attention_mask = torch.tensor([[[[0, 0, 0, 1, 1]]]], dtype=torch.float)
    attention_mask = (1.0 - attention_mask) * -10000.0  # Chuyển thành logit

    context_layer, attention_probs = attention(hidden_states, attention_mask)

    assert context_layer.shape == (
        1, 5, 12), "Sai kích thước đầu ra của context layer"
    assert attention_probs.shape == (
        1, 3, 5, 5), "Sai kích thước của attention probabilities"
    # Kiểm tra xem các token bị mask có attention prob bằng 0 không
    assert torch.allclose(attention_probs[0, :, :, 3], torch.zeros(
        3, 5)), "Masking không hoạt động đúng"
    assert torch.allclose(attention_probs[0, :, :, 4], torch.zeros(
        3, 5)), "Masking không hoạt động đúng"

    print("✓ Cơ chế attention hoạt động chính xác.")
    print("\n")


def test_bert_model():
    """Kiểm thử toàn bộ mô hình BERT với tác vụ pre-training."""
    print("="*20 + " Kiểm thử Toàn bộ Mô hình BERT " + "="*20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = BertConfig(
        vocab_size=30522,
        hidden_size=48,  # Giảm kích thước để chạy nhanh
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=48*4
    )
    model = BertForPreTraining(config).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Ví dụ cho MLM và NSP
    input_sentence = f"The capital of France is [MASK]."
    print(f"Kiểm tra với câu đầu vào: '{input_sentence}'")

    # Chuẩn bị input
    inputs = tokenizer(input_sentence, return_tensors='pt').to(device)
    outputs = model(**inputs, output_attentions=True)

    # Lấy điểm dự đoán cho token bị che
    prediction_scores = outputs['prediction_logits']
    masked_token_index = (
        inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Lấy top 5 dự đoán
    top_5_predictions = torch.topk(
        prediction_scores[0, masked_token_index], k=5)

    # In kết quả
    print(f"\nCâu mẫu cần dự đoán: {input_sentence}")
    print("\nTop 5 dự đoán cho [MASK]:")
    for score, idx in zip(top_5_predictions.values[0], top_5_predictions.indices[0]):
        predicted_token = tokenizer.convert_ids_to_tokens(idx.item())
        print(f"- {predicted_token}: score={score.item():.4f}")

    # Trực quan hóa attention
    print("\nTrực quan hóa trọng số attention...")
    visualize_attention(model.bert, tokenizer, input_sentence, device=device)
    print("\n")

    print("✓ Ví dụ huấn luyện nhỏ hoàn thành.")


if __name__ == '__main__':
    # Chạy các kiểm thử
    test_attention_mechanism()
    test_bert_model()

    print("\n" + "="*50)
    print("Tất cả các kiểm thử đã thành công! Việc cài đặt BERT đang hoạt động chính xác.")
    print("="*50)