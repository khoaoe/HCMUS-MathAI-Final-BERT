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
# tương ứng với các ký hiệu trong bài báo (Mục 3, "Model Architecture"):
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
    attention_probs_dropout_prob: float = 0.1   # Dropout cho attention
    max_position_embeddings: int = 512          # Độ dài chuỗi tối đa
    # Kích thước từ vựng của loại token (segment A/B)
    type_vocab_size: int = 2
    initializer_range: float = 0.02             # Độ lệch chuẩn khởi tạo trọng số
    layer_norm_eps: float = 1e-12               # Epsilon cho Layer Norm

    def __post_init__(self):
        # Đảm bảo kích thước ẩn có thể chia hết cho số lượng head
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) phải chia hết cho num_attention_heads ({self.num_attention_heads})"

# ========================= Cơ chế Attention (Attentions) =========================


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention - Cơ chế attention cốt lõi của Transformer.
    Công thức được mô tả trong bài báo "Attention Is All You Need":
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

    Việc chia cho sqrt(d_k) giúp ổn định gradient khi d_k có giá trị lớn.
    """

    def __init__(self, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,     # [batch, n_heads, seq_len, d_k]
        key: torch.Tensor,       # [batch, n_heads, seq_len, d_k]
        value: torch.Tensor,     # [batch, n_heads, seq_len, d_v]
        # mask để che đi các vị trí không cần attend
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: Tensor Query
            key: Tensor Key
            value: Tensor Value
            mask: Attention mask (1 cho vị trí attend, 0 cho vị trí bị che)
            return_attention: Cờ để quyết định có trả về trọng số attention hay không
        """
        batch_size, n_heads, seq_len, d_k = query.size()

        # Tính điểm attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / \
            (math.sqrt(d_k) * self.temperature)

        # Áp dụng mask nếu có
        if mask is not None:
            # Xử lý mask đúng cách cho attention matrix
            if mask.dim() == 2:  # [batch, seq_len] - padding mask
                # Chuyển thành causal mask [batch, seq_len, seq_len]
                batch_size, seq_len = mask.size()
                # Tạo attention mask: có thể attend nếu cả query và key đều không phải padding
                mask = mask.unsqueeze(1).expand(-1, seq_len, -1) * mask.unsqueeze(2).expand(-1, -1, seq_len)
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            elif mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            
            # Validation mask shape
            expected_shape = (scores.size(0), 1, scores.size(2), scores.size(3))
            if mask.shape != expected_shape:
                raise ValueError(f"Mask shape {mask.shape} không khớp với expected {expected_shape}")

            # Điền các vị trí bị mask bằng giá trị rất nhỏ (-inf) để sau softmax sẽ thành 0
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, fill_value)

        # Áp dụng softmax để có được xác suất attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Áp dụng attention lên value
        output = torch.matmul(attn_weights, value)

        if return_attention:
            return output, attn_weights
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention cho phép mô hình cùng lúc chú ý đến thông tin
    từ các không gian biểu diễn (representation subspaces) khác nhau.
    Đây là thành phần chính trong khối Transformer của BERT.

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    trong đó head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Kích thước của key
        self.d_v = d_model // n_heads  # Kích thước của value

        # Các phép chiếu tuyến tính (linear projections) cho Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)  # Phép chiếu đầu ra

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        # Khởi tạo Xavier uniform
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] hoặc [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len] (nếu return_attention=True)
        """
        batch_size, seq_len, _ = query.size()

        # 1. Chiếu tuyến tính và chia thành các head
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, seq_len,
                                 self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len,
                               self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len,
                                 self.n_heads, self.d_v).transpose(1, 2)

        # 2. Áp dụng attention trên tất cả các head
        if return_attention:
            attn_output, attn_weights = self.attention(
                Q, K, V, mask=mask, return_attention=True)
        else:
            attn_output = self.attention(
                Q, K, V, mask=mask, return_attention=False)
            attn_weights = None

        # 3. Ghép (concat) các head lại và áp dụng phép chiếu tuyến tính cuối cùng
        # [batch, n_heads, seq_len, d_v] -> [batch, seq_len, n_heads, d_v] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 4. Phép chiếu tuyến tính cuối cùng
        output = self.w_o(attn_output)
        output = self.dropout(output)

        if return_attention:
            return output, attn_weights
        return output

# ========================= Mạng Feed-Forward (Feed-Forward Network) =========================


class PositionwiseFeedForward(nn.Module):
    """
    Mạng Feed-Forward theo từng vị trí (Position-wise Feed-Forward Network).
    Đây là thành phần thứ hai trong một khối Transformer.
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    BERT sử dụng hàm kích hoạt GELU thay vì ReLU.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Hàm kích hoạt
        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation: str):
        """Lấy hàm kích hoạt theo tên"""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise ValueError(f"Hàm kích hoạt '{activation}' không được hỗ trợ")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ========================= Khối Transformer (Transformer Block) =========================


class TransformerBlock(nn.Module):
    """
    Một khối Transformer hoàn chỉnh. Trong BERT, các khối này được xếp chồng lên nhau.
    Mỗi khối bao gồm:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Cả hai đều có kết nối phần dư (residual connections) và chuẩn hóa tầng (layer normalization).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Các tầng con
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Chuẩn hóa tầng
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] hoặc None
            return_attention: Cờ để quyết định có trả về trọng số attention hay không
        """
        # Tầng self-attention với kết nối phần dư và chuẩn hóa
        if return_attention:
            attn_output, attn_weights = self.attention(
                x, x, x, mask, return_attention=True)
        else:
            attn_output = self.attention(x, x, x, mask, return_attention=False)
            attn_weights = None

        x = self.norm1(x + self.dropout(attn_output))

        # Tầng feed-forward với kết nối phần dư và chuẩn hóa
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        if return_attention:
            return x, attn_weights
        return x

# ========================= Embeddings =========================


class BertEmbeddings(nn.Module):
    """
    Tạo ra biểu diễn đầu vào cho BERT.
    Theo Hình 2 trong bài báo, embedding đầu vào là tổng của 3 thành phần:
    1. Token embeddings: Biểu diễn của từ/token.
    2. Position embeddings: Biểu diễn vị trí của token trong chuỗi (BERT học được).
    3. Token type (Segment) embeddings: Phân biệt câu A và câu B.

    Sau khi cộng, đầu ra được chuẩn hóa (LayerNorm) và áp dụng dropout.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        # Padding_idx=0 chỉ định rằng token padding sẽ có vector embedding là zero.
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Tạo buffer cho position_ids để không được coi là tham số của mô hình
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - ID của các token
            token_type_ids: [batch_size, seq_len] - ID của segment (0 hoặc 1)
            position_ids: [batch_size, seq_len] - ID của vị trí
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        seq_length = input_ids.size(1)

        # Lấy position IDs
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Lấy token type IDs (mặc định là 0 nếu không được cung cấp)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Lấy các embeddings tương ứng
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Cộng 3 loại embedding lại với nhau
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # Áp dụng LayerNorm và dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

# ========================= Bộ mã hóa BERT (BERT Encoder) =========================


class BertEncoder(nn.Module):
    """
    Bộ mã hóa của BERT, bao gồm một chồng các khối Transformer (TransformerBlock).
    Đây chính là "multi-layer bidirectional Transformer encoder" được nhắc đến trong bài báo.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                d_ff=config.intermediate_size,
                dropout=config.hidden_dropout_prob
            ) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: Trạng thái ẩn đầu vào [batch_size, seq_len, hidden_size]
            attention_mask: Mask cho attention [batch_size, seq_len]
            output_attentions: Cờ trả về trọng số attention
            output_hidden_states: Cờ trả về tất cả các trạng thái ẩn
        Returns:
            Một dictionary chứa:
                - last_hidden_state: Trạng thái ẩn của tầng cuối cùng
                - hidden_states: Danh sách các trạng thái ẩn (nếu output_hidden_states=True)
                - attentions: Danh sách các trọng số attention (nếu output_attentions=True)
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                hidden_states,
                mask=attention_mask,
                return_attention=output_attentions
            )

            if output_attentions:
                hidden_states, attn_weights = layer_outputs
                all_attentions.append(attn_weights)
            else:
                hidden_states = layer_outputs

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = {'last_hidden_state': hidden_states}
        if output_hidden_states:
            outputs['hidden_states'] = all_hidden_states
        if output_attentions:
            outputs['attentions'] = all_attentions

        return outputs


# ========================= BERT Pooler =========================
class BertPooler(nn.Module):
    """
    Lấy biểu diễn của token [CLS] và biến đổi nó để dùng cho các tác vụ phân loại câu.
    Trong bài báo, đây là vector 'C' (Mục 3.1).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Trạng thái ẩn từ tầng cuối của encoder [batch_size, seq_len, hidden_size]
        Returns:
            pooled_output: Biểu diễn tổng hợp của chuỗi [batch_size, hidden_size]
        """
        # Lấy trạng thái ẩn của token đầu tiên ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# ========================= Các đầu dự đoán (Prediction Heads) =========================


class BertPredictionHeadTransform(nn.Module):
    """
    Một tầng biến đổi (dense -> gelu -> layernorm) được áp dụng trước đầu dự đoán MLM.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = F.gelu
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Đầu dự đoán cho tác vụ Masked Language Model (MLM).
    Nó dự đoán token gốc từ biểu diễn ẩn của các token bị che.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Tầng decoder để chiếu từ hidden_size về vocab_size.
        # Trọng số của decoder được chia sẻ với ma trận word_embeddings (weight tying).
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    """
    Kết hợp hai đầu dự đoán cho hai tác vụ pre-training:
    1. Masked Language Modeling (MLM).
    2. Next Sentence Prediction (NSP).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(
            config.hidden_size, 2)  # 2 class: IsNext, NotNext

    def forward(
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_output: Đầu ra của encoder cho toàn chuỗi [batch_size, seq_len, hidden_size]
            pooled_output: Đầu ra của pooler (từ token [CLS]) [batch_size, hidden_size]
        Returns:
            prediction_scores: Logits cho tác vụ MLM [batch_size, seq_len, vocab_size]
            seq_relationship_score: Logits cho tác vụ NSP [batch_size, 2]
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

# ========================= Mô hình BERT (BERT Model) =========================


class BertModel(nn.Module):
    """
    Mô hình BERT hoàn chỉnh, tích hợp các thành phần:
    Embeddings -> Encoder -> Pooler.
    Đây là mô hình lõi, có thể được sử dụng cho pre-training hoặc fine-tuning.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # Khởi tạo trọng số
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Khởi tạo trọng số cho các tầng theo cấu hình"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Luồng xử lý chính của mô hình BERT.
        """
        # Tạo attention mask nếu không được cung cấp
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 1. Lấy embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        # 2. Đưa qua encoder
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs['last_hidden_state']

        # 3. Đưa qua pooler
        pooled_output = self.pooler(sequence_output)

        outputs = {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
        }
        if 'hidden_states' in encoder_outputs:
            outputs['hidden_states'] = encoder_outputs['hidden_states']
        if 'attentions' in encoder_outputs:
            outputs['attentions'] = encoder_outputs['attentions']

        return outputs

# ========================= Mô hình BERT cho Pre-training =========================


class BertForPreTraining(nn.Module):
    """
    Mô hình BERT với các đầu pre-training (MLM và NSP).
    Mô hình này được sử dụng để huấn luyện BERT từ dữ liệu không nhãn (unlabeled text).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Chia sẻ trọng số (weight tying) giữa input embeddings và output decoder
        self.tie_weights()

    def tie_weights(self):
        """Chia sẻ trọng số giữa word_embeddings và decoder của MLM head."""
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,         # Nhãn cho MLM
        next_sentence_label: Optional[torch.Tensor] = None,  # Nhãn cho NSP
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Luồng xử lý và tính toán loss cho quá trình pre-training.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']

        # Lấy điểm dự đoán từ các đầu pre-training
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        total_loss = None
        mlm_loss = None
        nsp_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()  # Bỏ qua index -100

            # Tính loss cho MLM
            mlm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1)
            )

            # Tính loss cho NSP
            nsp_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )

            # Tổng loss là tổng của hai loss trên
            total_loss = mlm_loss + nsp_loss

        return {
            'loss': total_loss,
            'mlm_loss': mlm_loss,
            'nsp_loss': nsp_loss,
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score,
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions')
        }

# ========================= Mô hình BERT cho Phân loại Chuỗi =========================


class BertForSequenceClassification(nn.Module):
    """
    Mô hình BERT cho tác vụ phân loại chuỗi (fine-tuning).
    Ví dụ: phân loại cảm xúc, phân loại chủ đề.
    Mô hình này thêm một tầng tuyến tính (linear layer) đơn giản vào sau BertPooler.
    """

    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None  # Nhãn phân loại
    ) -> Dict[str, torch.Tensor]:
        """
        Luồng xử lý và tính toán loss cho tác vụ phân loại.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['last_hidden_state'],
            'pooler_output': outputs['pooler_output']
        }

# ========================= Cài đặt Dataset =========================


class BertDataset(Dataset):
    """
    Dataset cho quá trình pre-training BERT với hai tác vụ MLM và NSP.
    Lớp này thực hiện việc tạo các cặp câu và chiến lược che token (masking).
    Chiến lược masking (Mục 3.1):
    - 80% thời gian: Thay thế bằng token [MASK]
    - 10% thời gian: Thay thế bằng một token ngẫu nhiên
    - 10% thời gian: Giữ nguyên token gốc
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

        # Tiền xử lý văn bản thành các câu trong các tài liệu
        self.documents = self._preprocess_texts()

    def _preprocess_texts(self) -> List[List[str]]:
        """Tách văn bản thành các câu"""
        documents = []
        for text in self.texts:
            # Tách câu đơn giản bằng dấu chấm
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                documents.append(sentences)
        return documents

    def _get_random_sentence(self, exclude_doc_idx: int) -> str:
        """Lấy một câu ngẫu nhiên từ một tài liệu khác"""
        if len(self.documents) == 1:
            return ""
        doc_idx = random.choice(
            [i for i in range(len(self.documents)) if i != exclude_doc_idx])
        if self.documents[doc_idx]:
            return random.choice(self.documents[doc_idx])
        return ""

    def _create_training_instance(self, doc_idx: int) -> Tuple[str, str, int]:
        """Tạo một mẫu huấn luyện với câu A, câu B và nhãn NSP"""
        document = self.documents[doc_idx]
        sent_idx_a = random.randint(0, len(document) - 1)
        sent_a = document[sent_idx_a]

        # Tạo câu B và nhãn NSP (is_next)
        if random.random() < 0.5 and sent_idx_a < len(document) - 1:
            # 50% là câu tiếp theo (ví dụ dương)
            sent_b = document[sent_idx_a + 1]
            is_next = 1
        else:
            # 50% là câu ngẫu nhiên (ví dụ âm)
            sent_b = self._get_random_sentence(doc_idx)
            is_next = 0

        return sent_a, sent_b, is_next

    def _truncate_seq_pair(self, tokens_a: List[int], tokens_b: List[int], max_length: int):
        """Cắt bớt cặp chuỗi để vừa với độ dài tối đa"""
        # -3 cho các token [CLS], [SEP], [SEP]
        while len(tokens_a) + len(tokens_b) > max_length - 3:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _create_masked_lm_predictions(
        self,
        tokens: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Tạo các dự đoán cho masked language model.
        Returns:
            output_tokens: Các token đã áp dụng masking.
            output_labels: Các token gốc ở vị trí bị mask (-100 cho vị trí không bị mask).
        """
        output_tokens = tokens.copy()
        # -100 được CrossEntropyLoss bỏ qua khi tính loss
        output_labels = [-100] * len(tokens)

        # Lấy các vị trí có thể mask (không phải token đặc biệt)
        candidate_indices = [i for i, token in enumerate(tokens) if token not in
                             [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]]

        # Chọn ngẫu nhiên 15% các vị trí để mask
        random.shuffle(candidate_indices)
        num_to_mask = max(
            1, int(len(candidate_indices) * self.mlm_probability))
        mask_indices = candidate_indices[:num_to_mask]

        for idx in mask_indices:
            # Lưu nhãn (token gốc)
            output_labels[idx] = tokens[idx]

            # 80% thay bằng [MASK]
            if random.random() < 0.8:
                output_tokens[idx] = self.tokenizer.mask_token_id
            # 10% thay bằng token ngẫu nhiên
            elif random.random() < 0.5:  # (0.5 của 20% còn lại là 10%)
                output_tokens[idx] = random.randint(
                    0, self.tokenizer.vocab_size - 1)
            # 10% giữ nguyên token gốc

        return output_tokens, output_labels

    def __len__(self) -> int:
        # Tạo nhiều mẫu huấn luyện từ mỗi tài liệu
        return len(self.documents) * 10

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc_idx = idx % len(self.documents)

        # 1. Tạo cặp câu cho NSP
        sent_a, sent_b, is_next = self._create_training_instance(doc_idx)

        # 2. Tokenize và cắt bớt
        tokens_a = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sent_a))
        tokens_b = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sent_b)) if sent_b else []
        self._truncate_seq_pair(tokens_a, tokens_b, self.max_length)

        # 3. Xây dựng chuỗi đầu vào: [CLS] A [SEP] B [SEP]
        tokens = [self.tokenizer.cls_token_id] + \
            tokens_a + [self.tokenizer.sep_token_id]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [self.tokenizer.sep_token_id]
            segment_ids += [1] * (len(tokens_b) + 1)

        # 4. Tạo MLM predictions
        masked_tokens, mlm_labels = self._create_masked_lm_predictions(tokens)

        # 5. Padding
        attention_mask = [1] * len(masked_tokens)
        padding_length = self.max_length - len(masked_tokens)
        masked_tokens.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        segment_ids.extend([0] * padding_length)
        mlm_labels.extend([-100] * padding_length)

        return {
            'input_ids': torch.tensor(masked_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            'labels': torch.tensor(mlm_labels, dtype=torch.long),
            'next_sentence_label': torch.tensor(is_next, dtype=torch.long)
        }

# ========================= Các tiện ích Huấn luyện (Training Utilities) =========================


class BertTrainer:
    """Lớp tiện ích để thực hiện quá trình huấn luyện BERT"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 10000,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False
    ):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        # Optimizer AdamW, không áp dụng weight decay cho bias và LayerNorm
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate)

        # Scheduler để điều chỉnh learning rate (warmup rồi giảm tuyến tính)
        # Giả sử 10 epochs
        total_steps = len(train_dataloader) // gradient_accumulation_steps * 10
        self.scheduler = self._get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps)

        # Hỗ trợ huấn luyện với độ chính xác hỗn hợp (mixed precision)
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision and self.device == 'cuda' else None

        # Lưu lịch sử huấn luyện
        self.train_losses, self.val_losses, self.learning_rates = [], [], []

    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Tạo scheduler với learning rate tăng tuyến tính trong warmup, sau đó giảm tuyến tính"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch(self) -> Dict[str, float]:
        """Huấn luyện trong một epoch"""
        self.model.train()
        total_loss, total_mlm_loss, total_nsp_loss = 0, 0, 0
        from tqdm import tqdm
        progress_bar = tqdm(self.train_dataloader, desc="Đang huấn luyện")

        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Trích xuất kiểu thiết bị dưới dạng chuỗi ('cuda' hoặc 'cpu')
            device_type_str = self.device.type if isinstance(self.device, torch.device) else self.device

            # Forward pass với mixed precision
            with torch.amp.autocast(device_type=device_type_str, dtype=torch.float16, enabled=self.scaler is not None):
                outputs = self.model(**batch)
                loss = outputs['loss']
            if loss is None:
                continue

            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Cập nhật trọng số sau mỗi `gradient_accumulation_steps`
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Cắt gradient để tránh bùng nổ gradient
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

                # Cập nhật optimizer và scheduler
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Ghi nhận loss
            total_loss += outputs['loss'].item()
            if outputs['mlm_loss'] is not None:
                total_mlm_loss += outputs['mlm_loss'].item()
            if outputs['nsp_loss'] is not None:
                total_nsp_loss += outputs['nsp_loss'].item()
            progress_bar.set_postfix(
                {'loss': f"{outputs['loss'].item():.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})
            self.learning_rates.append(self.scheduler.get_last_lr()[0])

        num_batches = len(self.train_dataloader)
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return {
            'loss': avg_loss,
            'mlm_loss': total_mlm_loss / num_batches,
            'nsp_loss': total_nsp_loss / num_batches
        }

    def validate(self) -> Dict[str, float]:
        """Đánh giá mô hình trên tập validation"""
        if self.val_dataloader is None:
            return {}
        self.model.eval()
        total_loss, total_mlm_loss, total_nsp_loss = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Đang đánh giá"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                if outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()
                if outputs['mlm_loss'] is not None:
                    total_mlm_loss += outputs['mlm_loss'].item()
                if outputs['nsp_loss'] is not None:
                    total_nsp_loss += outputs['nsp_loss'].item()

        num_batches = len(self.val_dataloader)
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return {
            'loss': avg_loss,
            'mlm_loss': total_mlm_loss / num_batches,
            'nsp_loss': total_nsp_loss / num_batches
        }

    def train(self, num_epochs: int, save_dir: Optional[str] = None):
        """Vòng lặp huấn luyện chính"""
        print(
            f"Bắt đầu huấn luyện {num_epochs} epochs trên thiết bị: {self.device}")
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"{'='*50}\nEpoch {epoch + 1}/{num_epochs}\n{'='*50}")
            train_metrics = self.train_epoch()
            print(
                f"\nKết quả huấn luyện: Loss={train_metrics['loss']:.4f}, MLM Loss={train_metrics['mlm_loss']:.4f}, NSP Loss={train_metrics['nsp_loss']:.4f}")
            val_metrics = self.validate()
            if val_metrics:
                print(
                    f"Kết quả đánh giá: Loss={val_metrics['loss']:.4f}, MLM Loss={val_metrics['mlm_loss']:.4f}, NSP Loss={val_metrics['nsp_loss']:.4f}")
                if save_dir and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_dir, epoch + 1, is_best=True)
            if save_dir:
                self.save_checkpoint(save_dir, epoch + 1)

    def save_checkpoint(self, save_dir: str, epoch: int, is_best: bool = False):
        """Lưu checkpoint của mô hình"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir, 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
        }, path)
        print(f"Đã lưu checkpoint: {path}")

    def plot_training_history(self):
        """Vẽ biểu đồ lịch sử huấn luyện"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set(xlabel='Epoch', ylabel='Loss',
                    title='Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(self.learning_rates)
        axes[1].set(xlabel='Steps', ylabel='Learning Rate',
                    title='Learning Rate Schedule')
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()


# ========================= Trực quan hóa (Visualization) =========================
def visualize_attention(
    model: BertModel,
    tokenizer: BertTokenizer,
    text: str,
    layer_idx: int = -1,
    head_idx: int = 0,
    device: str = 'cpu'
):
    """Trực quan hóa trọng số attention cho một đoạn văn bản"""
    inputs = tokenizer.encode_plus(
        text, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention = outputs['attentions'][layer_idx][0, head_idx].cpu() 
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu()) 

    plt.figure(figsize=(10, 8))
    # Chuyển attention về numpy để vẽ
    sns.heatmap(attention.numpy(), xticklabels=tokens,
                yticklabels=tokens, cmap='Blues')
    plt.title(f'Trọng số Attention - Tầng {layer_idx}, Head {head_idx}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ========================= Kiểm thử (Testing) =========================


def test_attention_mechanism():
    """Kiểm tra cơ chế attention"""
    print("Kiểm tra cơ chế Attention...")
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    x = torch.randn(2, 5, 64)  # batch_size=2, seq_len=5, d_model=64
    output, _ = mha(x, x, x, return_attention=True)
    print(f"Kích thước đầu vào: {x.shape}")
    print(f"Kích thước đầu ra: {output.shape}")
    assert output.shape == x.shape, "Lỗi: Kích thước đầu ra không khớp!"
    print("✓ Kiểm tra cơ chế attention thành công!")


def test_bert_model():
    """Kiểm tra toàn bộ mô hình BERT"""
    print("Kiểm tra mô hình BERT...")
    config = BertConfig(vocab_size=1000, hidden_size=128,
                        num_hidden_layers=2, num_attention_heads=4, intermediate_size=512)
    model = BertForPreTraining(config)

    # Tạo dữ liệu giả
    input_ids = torch.randint(0, config.vocab_size, (4, 20))
    labels = torch.full_like(input_ids, -100)
    # Tạo một mask duy nhất để tái sử dụng
    mask = torch.rand_like(input_ids.float()) < 0.15
    # Sử dụng cùng một mask cho cả hai bên của phép gán
    labels[mask] = input_ids[mask]
    # labels[torch.rand_like(input_ids.float()) < 0.15] = input_ids[torch.rand_like(input_ids.float()) < 0.15]
    next_sentence_label = torch.randint(0, 2, (4,))

    outputs = model(input_ids=input_ids, labels=labels,
                    next_sentence_label=next_sentence_label)

    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Kích thước logits MLM: {outputs['prediction_logits'].shape}")
    print(f"Kích thước logits NSP: {outputs['seq_relationship_logits'].shape}")
    print("✓ Kiểm tra mô hình BERT thành công!")


def quick_demo(device: str = 'cpu', input_sentence: str = "BERT is a [MASK]."):
    print("="*50)
    print("Bắt đầu minh họa huấn luyện nhỏ...")
    print("="*50)

    print(f"\nSử dụng thiết bị: {device}")

    # 1. Khởi tạo Tokenizer
    print("\n1. Khởi tạo Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("\n")

    # 2. Tạo dữ liệu giả
    print("\n2. Tạo dữ liệu giả...")
    sample_texts = [
        "BERT is a model for language understanding. It was developed by Google.",
        "The model is based on the Transformer architecture. It uses bidirectional self-attention.",
        "Pre-training is done on a large corpus. Fine-tuning is for specific tasks.",
        "There are two pre-training tasks. Masked LM and Next Sentence Prediction.",
        "This implementation is a simplified version. It helps to understand the core concepts.",
        "BERT can be used for various NLP tasks. Examples include sentiment analysis and question answering.",
        "The model can handle long sequences. It uses position embeddings to maintain order.",
        "BERT has achieved state-of-the-art results. It is widely used in the NLP community.",
        "The model is trained on large datasets. It requires significant computational resources.",
        "BERT's architecture allows for deep understanding. It captures context from both directions."
    ]
    print(f"Tổng số câu mẫu: {len(sample_texts)}")
    print("\n")

    # 3. Tạo Dataset và DataLoader
    print("\n3. Tạo Dataset và DataLoader...")
    dataset = BertDataset(texts=sample_texts,
                          tokenizer=tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("Tổng số mẫu trong Dataset:", len(dataset))
    print("Số lượng batch trong DataLoader:", len(dataloader))
    print("\n")

    # 4. Khởi tạo mô hình và cấu hình
    print("\n4. Khởi tạo mô hình BERT...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=64
    )
    model = BertForPreTraining(config)
    print(f"Cấu hình mô hình: {config}")
    print(
        f"Kích thước mô hình: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    print("\n")

    # 5. Khởi tạo Trainer
    print("\n5. Khởi tạo Trainer...")
    trainer = BertTrainer(
        model=model,
        train_dataloader=dataloader,
        learning_rate=5e-5,
        warmup_steps=100,
        gradient_accumulation_steps=1,
        device=device,
        mixed_precision=torch.cuda.is_available()  # Sử dụng mixed precision nếu có GPU
    )
    print("Thiết bị huấn luyện:", trainer.device)
    print("\n")

    # 6. Huấn luyện trong 1 epoch
    print("\n6. Bắt đầu huấn luyện mô hình...")
    trainer.train(num_epochs=3)
    print("Huấn luyện hoàn thành!")
    print("\n")

    # 7. Vẽ biểu đồ
    print("\n7. Vẽ biểu đồ lịch sử huấn luyện...")
    trainer.plot_training_history()
    print("\n")

    # 8. Testing...
    print("\n8. Testing...")
    # Tokenize input
    inputs = tokenizer(input_sentence, return_tensors='pt')

    # Chuyển các tensor đầu vào sang cùng thiết bị với model (quan trọng!)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get prediction scores for masked token
    prediction_scores = outputs['prediction_logits']
    masked_token_index = (
        inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Get top 5 predictions
    top_5_predictions = torch.topk(
        prediction_scores[0, masked_token_index], k=5)

    # Print results
    print(f"Câu mẫu cần dự đoán: {input_sentence}")
    print("\nTop 5 dự đoán cho [MASK]:")
    for score, idx in zip(top_5_predictions.values[0], top_5_predictions.indices[0]):
        predicted_token = tokenizer.convert_ids_to_tokens(idx.item())
        print(f"- {predicted_token}: score={score.item():.4f}")

    # Visualize attention for the input sentence
    print("\nVisualize trọng số attention...")
    visualize_attention(model.bert, tokenizer, input_sentence, device=device)
    print("\n")

    print("\n✓ Ví dụ huấn luyện nhỏ hoàn thành.")


if __name__ == '__main__':
    # Chạy các kiểm thử
    test_attention_mechanism()
    test_bert_model()

    print("\n" + "="*50)
    print("Tất cả các kiểm thử đã thành công! Việc cài đặt BERT đang hoạt động chính xác.")
    print("="*50)

    # Minh họa về tạo dữ liệu và huấn luyện
    try:
        quick_demo()

    except Exception as e:
        print(f"\nLỗi trong quá trình chạy ví dụ huấn luyện: {e}")
        print("Vui lòng đảm bảo đã cài đặt các thư viện cần thiết (torch, transformers, tqdm, matplotlib, seaborn).")
