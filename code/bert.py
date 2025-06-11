# ========================= Phần Import Thư viện =========================
# Phần này định nghĩa tất cả các thư viện và module cần thiết cho việc xây dựng,
# huấn luyện và kiểm thử mô hình BERT. Mỗi import đều có một mục đích cụ thể.

# --- PyTorch Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
# torch, nn, và nn.functional là nền tảng của PyTorch để xây dựng mạng neural.
# - torch: Cung cấp các cấu trúc dữ liệu tensor và các phép toán toán học.
# - nn: Chứa các lớp (layers), hàm lỗi (loss functions), và các khối xây dựng
#   cơ bản cho mạng neural (ví dụ: nn.Module, nn.Linear, nn.Embedding).
# - F: Chứa các hàm chức năng như hàm kích hoạt (ReLU, GeLU) mà có thể
#   áp dụng trực tiếp.

# --- Thư viện Toán học và Tiện ích Python ---
import math  # Dùng cho các phép toán, cụ thể là math.sqrt để chia tỷ lệ trong attention.
import random  # Dùng để tạo tính ngẫu nhiên trong việc tạo dữ liệu cho MLM và NSP.
from typing import Dict, List, Tuple, Optional, Union  # Cung cấp type hints để code rõ ràng hơn.
import time  # Dùng để đo lường thời gian (hiện chưa dùng nhưng hữu ích).
from dataclasses import dataclass  # Cung cấp decorator @dataclass để tạo các lớp cấu hình gọn gàng.

# --- Thư viện Hugging Face Transformers ---
from transformers import BertTokenizer
# - BertTokenizer: Công cụ để chuyển đổi văn bản thành các ID mà BERT hiểu được (tokenization).

# --- Tiện ích Xử lý Dữ liệu của PyTorch ---
from torch.utils.data import Dataset, DataLoader
# - Dataset: Một lớp trừu tượng để tạo các bộ dữ liệu tùy chỉnh (ví dụ: PretrainingDataset).
# - DataLoader: Giúp tạo các batch dữ liệu, xáo trộn (shuffle) và tải dữ liệu song song.

# --- Thư viện Trực quan hóa và Theo dõi Tiến trình ---
import matplotlib.pyplot as plt  # Thư viện phổ biến để vẽ biểu đồ và đồ thị.
import seaborn as sns  # Xây dựng trên matplotlib, giúp tạo các biểu đồ thống kê đẹp hơn.
from tqdm import tqdm  # Tạo một thanh tiến trình (progress bar) trực quan cho các vòng lặp,
                       # rất hữu ích khi theo dõi quá trình huấn luyện.

# ========================= Cấu hình (Configuration) =========================
@dataclass
class BertConfig:
    """
    Lớp cấu hình (Configuration Class) cho mô hình BERT.

    Lớp này lưu trữ tất cả các siêu tham số (hyperparameter) của mô hình.
    Việc gom chúng vào một nơi giúp mã nguồn sạch sẽ, dễ quản lý và dễ dàng
    thay đổi cấu hình để thử nghiệm (ví dụ: BERT-base vs. BERT-large).
    Các tham số tương ứng với các ký hiệu trong bài báo gốc (Mục 3):
    - num_hidden_layers: Số tầng Transformer (L)
    - hidden_size: Kích thước ẩn (H)
    - num_attention_heads: Số lượng attention head (A)
    - intermediate_size: Kích thước tầng feed-forward (4*H)
    """
    vocab_size: int = 30522                     # Kích thước bộ từ vựng của tokenizer
    hidden_size: int = 768                      # Kích thước của các vector biểu diễn
    num_hidden_layers: int = 12                 # Số khối Transformer Encoder xếp chồng lên nhau
    num_attention_heads: int = 12               # Số lượng "đầu" attention trong Multi-Head Attention
    intermediate_size: int = 3072               # Kích thước của tầng ẩn trong Feed-Forward Network
    hidden_dropout_prob: float = 0.1            # Tỷ lệ dropout cho các tầng fully-connected
    attention_probs_dropout_prob: float = 0.1   # Tỷ lệ dropout cho các trọng số attention
    max_position_embeddings: int = 512          # Độ dài tối đa của một chuỗi đầu vào
    type_vocab_size: int = 2                    # Số loại segment (câu A là 0, câu B là 1)
    initializer_range: float = 0.02             # Độ lệch chuẩn cho việc khởi tạo trọng số theo phân phối chuẩn


# ========================= Các Module con của BERT =========================

class BertEmbeddings(nn.Module):
    """
    Tầng Embedding của BERT.

    Chịu trách nhiệm chuyển đổi chuỗi ID đầu vào thành các vector biểu diễn.
    Theo bài báo, biểu diễn đầu vào của mỗi token được tạo ra bằng cách cộng
    tổng 3 loại embedding tương ứng:
    1. Token Embeddings: Biểu diễn cho từng token trong từ vựng.
    2. Segment Embeddings: Phân biệt giữa câu A và câu B (cần cho tác vụ NSP).
    3. Position Embeddings: Cung cấp thông tin về vị trí của token trong chuỗi.
       BERT học các embedding vị trí này thay vì dùng hàm sin/cos cố định.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        # Tạo lớp embedding cho token, với padding_idx=0 để vector của token [PAD] luôn là zero.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # Tạo lớp embedding cho vị trí, có thể học được.
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # Tạo lớp embedding cho loại segment (câu A/B).
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Layer Normalization: Chuẩn hóa các embedding sau khi cộng tổng.
        # Giúp ổn định quá trình huấn luyện. `eps` là một giá trị nhỏ để tránh chia cho 0.
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # Dropout: Một kỹ thuật điều chuẩn (regularization) để tránh overfitting.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Lấy độ dài chuỗi từ kích thước của input_ids.
        seq_length = input_ids.size(1)
        # Tạo ra một tensor chứa các ID vị trí (từ 0 đến seq_length - 1).
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # Mở rộng position_ids để có cùng kích thước batch với input_ids.
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Nếu token_type_ids không được cung cấp, mặc định tất cả đều là segment 0.
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Lấy các embedding tương ứng từ các lớp đã tạo.
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Cộng tổng 3 loại embedding - đây là cốt lõi của tầng embedding trong BERT.
        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        # Áp dụng Layer Normalization và Dropout.
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Lớp Multi-Head Self-Attention.

    Đây là thành phần cốt lõi của kiến trúc Transformer. Nó cho phép một token
    "nhìn" vào tất cả các token khác trong chuỗi và tính toán một trọng số "chú ý"
    để xác định mức độ quan trọng của các token khác đối với nó.
    Công thức Attention: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    Multi-Head: Thay vì tính attention một lần, BERT chia nhỏ vector biểu diễn
    thành nhiều "đầu" (head), mỗi đầu tính attention độc lập. Điều này cho phép
    mô hình học các loại quan hệ khác nhau giữa các từ.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        # Số lượng attention head (A).
        self.num_attention_heads = config.num_attention_heads
        # Kích thước của mỗi head. Phải đảm bảo hidden_size chia hết cho num_attention_heads.
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # Tổng kích thước của tất cả các head, bằng hidden_size.
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Các lớp Linear để tạo ra các ma trận Query, Key, Value từ hidden_states đầu vào.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Lớp Dropout để áp dụng lên các xác suất attention.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hàm này thay đổi hình dạng của tensor Q, K, V để phù hợp cho việc tính toán multi-head.
        Từ (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Chiếu hidden_states qua các lớp linear để tạo Q, K, V.
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Thay đổi hình dạng Q, K, V cho multi-head.
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Tính attention scores: Q @ K.T
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # Chia cho căn bậc hai của kích thước head (sqrt(d_k)).
        # Đây là bước "Scaled" trong "Scaled Dot-Product Attention", giúp ổn định gradient.
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Áp dụng attention mask để các token [PAD] không được chú ý đến.
            # Mask có giá trị rất nhỏ (-10000.0) sẽ trở thành gần 0 sau khi qua softmax.
            attention_scores = attention_scores + attention_mask

        # Chuẩn hóa attention scores thành xác suất bằng hàm softmax.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Tính context layer bằng cách nhân attention probs với V.
        context_layer = torch.matmul(attention_probs, value_layer)

        # Nối các head lại với nhau.
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Trả về context layer (đầu ra của attention) và attention probabilities (để trực quan hóa).
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """
    Lớp ouput sau Multi-Head Attention.

    Bao gồm một tầng linear, dropout, và một kết nối phần dư (residual connection)
    với Layer Normalization. Cấu trúc này (Add & Norm) là một phần tiêu chuẩn
    của kiến trúc Transformer.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states: là đầu ra từ attention (context_layer)
        # input_tensor: là đầu vào của attention (hidden_states ban đầu)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Kết nối phần dư (Residual Connection): cộng đầu vào ban đầu với đầu ra đã qua xử lý.
        # Giúp tránh vấn đề vanishing gradient trong các mạng sâu.
        # Sau đó áp dụng Layer Normalization.
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    Lớp này gói gọn toàn bộ khối Self-Attention,
    bao gồm cả MultiHeadSelfAttention và BertSelfOutput (phần Add & Norm).
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = MultiHeadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tính toán đầu ra của multi-head attention.
        self_outputs = self.self(hidden_states, attention_mask)
        # Đưa đầu ra này qua lớp output (Dense + Dropout + Add & Norm).
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    """
    Tầng Feed-Forward đầu tiên trong khối Transformer Encoder.
    Nó mở rộng chiều của vector biểu diễn (ví dụ từ 768 -> 3072).
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # BERT sử dụng hàm kích hoạt 'gelu' (Gaussian Error Linear Unit).
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    Tầng Feed-Forward thứ hai, thu hẹp chiều vector trở lại kích thước ban đầu.
    Cũng bao gồm một kết nối phần dư và Layer Normalization.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states: là đầu ra từ BertIntermediate
        # input_tensor: là đầu vào của BertIntermediate (tức là đầu ra của BertAttention)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Kết nối phần dư và chuẩn hóa.
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    Một khối Transformer Encoder Layer hoàn chỉnh.

    Mỗi khối này bao gồm hai thành phần chính:
    1. Một cơ chế Multi-Head Self-Attention.
    2. Một mạng Feed-Forward (Position-wise).
    Cả hai thành phần đều có kết nối phần dư và Layer Normalization.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Truyền hidden_states qua khối Attention.
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]

        # Truyền đầu ra của Attention qua khối Feed-Forward.
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    """
    Bộ mã hóa của BERT (BERT Encoder).

    Bao gồm việc xếp chồng nhiều lớp `BertLayer` lên nhau.
    Ví dụ: BERT-base có 12 lớp, BERT-large có 24 lớp.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        all_attentions = () if output_attentions else None
        # Lần lượt cho dữ liệu đi qua từng lớp của encoder.
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0] # Đầu ra của lớp này là đầu vào của lớp tiếp theo.
            if output_attentions:
                # Lưu lại trọng số attention của từng lớp nếu cần.
                all_attentions = all_attentions + (layer_outputs[1],)

        return {'last_hidden_state': hidden_states, 'attentions': all_attentions}


class BertPooler(nn.Module):
    """
    Lớp Pooler của BERT.

    Nhiệm vụ của nó là lấy vector biểu diễn của token đầu tiên ([CLS]) từ lớp cuối cùng
    của Encoder. Vector này được coi là biểu diễn tổng hợp cho toàn bộ câu và
    thường được sử dụng cho các tác vụ phân loại câu (như NSP hoặc phân loại cảm xúc).
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Lấy hidden state của token đầu tiên (vị trí 0).
        first_token_tensor = hidden_states[:, 0]
        # Cho qua một lớp linear và hàm kích hoạt Tanh.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """
    Một lớp biến đổi nhỏ (transform) trước khi dự đoán token cho tác vụ MLM.
    Nó bao gồm một lớp Linear, một hàm kích hoạt GELU, và LayerNorm.
    """
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
    Đầu ra (Prediction Head) cho tác vụ Masked Language Model (MLM).

    Nhận đầu ra từ encoder, biến đổi nó, và sau đó chiếu nó vào không gian từ vựng
    để dự đoán token bị che (masked) là gì.
    """
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # Tầng decoder này chiếu từ hidden_size ra vocab_size.
        # Trọng số của nó được chia sẻ (tied) với trọng số của tầng token embedding.
        # Đây là một kỹ thuật phổ biến giúp giảm số lượng tham số.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# ========================= Mô hình BERT Hoàn chỉnh =========================

class BertModel(nn.Module):
    """
    Mô hình BERT cơ bản, chỉ bao gồm Embedding, Encoder và Pooler.

    Lớp này trả về các biểu diễn ẩn (hidden representations) từ encoder.
    Nó không bao gồm các đầu ra cho tác vụ cụ thể (như MLM hay NSP),
    và thường được dùng làm nền tảng cho các mô hình khác.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        if attention_mask is None: attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)

        # Tạo attention mask mở rộng để broadcasting qua các head.
        # Từ (B, S) -> (B, 1, 1, S).
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # Chuyển mask thành giá trị -10000.0 (cho vị trí mask) và 0.0 (cho vị trí không mask).
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, output_attentions)
        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output)
        return {'last_hidden_state': sequence_output, 'pooler_output': pooled_output, 'attentions': encoder_outputs['attentions']}


class BertForPreTraining(nn.Module):
    """
    Mô hình BERT hoàn chỉnh cho việc Pre-training.

    Bao gồm mô hình BERT cơ bản và các đầu ra cho cả hai tác vụ:
    1. Masked Language Model (MLM): Dự đoán các token bị che.
    2. Next Sentence Prediction (NSP): Dự đoán xem hai câu có đi liền nhau không.
    Lớp này cũng tính toán tổng loss từ hai tác vụ trên.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        # Đầu ra cho tác vụ MLM.
        self.cls_mlm = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        # Đầu ra cho tác vụ NSP (phân loại 2 lớp).
        self.cls_nsp = nn.Linear(config.hidden_size, 2)
        self._init_weights()

    def _init_weights(self):
        # Hàm khởi tạo trọng số theo đúng chuẩn của BERT.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels=None, next_sentence_label=None, output_attentions=False):
        # Lấy đầu ra từ mô hình BERT cơ bản.
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_attentions)
        sequence_output = outputs['last_hidden_state'] # Biểu diễn của tất cả token
        pooled_output = outputs['pooler_output'] # Biểu diễn của [CLS]

        # Đưa các biểu diễn qua các đầu ra tác vụ tương ứng.
        prediction_scores = self.cls_mlm(sequence_output) # Logits cho MLM
        seq_relationship_score = self.cls_nsp(pooled_output) # Logits cho NSP

        total_loss = None
        # Nếu có nhãn, tính toán loss.
        if masked_lm_labels is not None and next_sentence_label is not None:
            # Dùng CrossEntropyLoss cho cả hai tác vụ.
            loss_fct = nn.CrossEntropyLoss()
            # Tính loss cho MLM, bỏ qua các token có nhãn -100.
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.bert.config.vocab_size), masked_lm_labels.view(-1))
            # Tính loss cho NSP.
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            # Tổng loss là tổng của hai loss trên.
            total_loss = masked_lm_loss + next_sentence_loss

        return {
            'loss': total_loss,
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score,
            'attentions': outputs['attentions']
        }


# ========================= Phần Huấn luyện và Kiểm thử =========================

# --- Bước 1: Chuẩn bị Dữ liệu và Lớp Dataset ---

# Một tập dữ liệu nhỏ để huấn luyện thử
small_corpus = [
    ("The man went to the store.", "He bought a gallon of milk."),
    ("She reads a book.", "The book is about dragons."),
    ("BERT is a powerful model.", "It was developed by Google."),
    ("An apple a day keeps the doctor away.", "The cat sat on the mat."),
    ("The dog is cute.", "The sky is blue."),
    ("A transformer is a type of model.", "This model uses an attention mechanism."),
    ("This is a language model.", "The model can understand text."),
    ("They built a new AI model.", "The model has many parameters."),
    ("This system uses a predictive model.", "The model forecasts future sales."),
    ("The new model is very efficient.", "It runs faster than the old one."),
    ("GPT-3 is a large language model.", "The model was trained by OpenAI."),
    ("This machine learning model is complex.", "The model requires a lot of data."),
    # --- Dữ liệu bổ sung để dạy về câu liên quan (NSP) ---
    ("I have a new car.", "The car is red."),
    ("The student studied for the exam.", "He passed with a high score."),
    ("We went to the restaurant last night.", "The food there was delicious."),
    ("She wrote a long letter.", "The letter was for her best friend."),
    ("The library is quiet.", "People are reading books."),
    # --- Dữ liệu bổ sung để dạy về câu không liên quan (NSP) ---
    ("The sun is shining today.", "My favorite color is green."),
    ("He drinks coffee every morning.", "History is an interesting subject."),
    ("The train arrived on time.", "Elephants are very large mammals."),
    ("She likes to play the piano.", "The ocean is very deep."),
    ("My computer is very fast.", "Birds can fly high in the sky.")
]
class PretrainingDataset(Dataset):
    """
    Lớp Dataset để xử lý dữ liệu cho cả 2 tác vụ pre-training: MLM và NSP.
    Kế thừa từ `torch.utils.data.Dataset`, cần cài đặt `__len__` và `__getitem__`.
    """
    def __init__(self, corpus, tokenizer, max_len=64):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Trả về số lượng mẫu trong dataset.
        return len(self.corpus)

    def __getitem__(self, idx):
        # Hàm này định nghĩa cách lấy một mẫu dữ liệu tại vị trí `idx`.

        # === Chuẩn bị cho tác vụ NSP ===
        sent_a, sent_b = self.corpus[idx]
        is_next = True
        # 50% xác suất chọn một câu ngẫu nhiên làm câu B.
        if random.random() < 0.5:
            rand_idx = random.randint(0, len(self.corpus) - 1)
            sent_b = self.corpus[rand_idx][1]
            is_next = False

        # Tokenize cặp câu, thêm [CLS], [SEP], padding và truncation.
        tokenized = self.tokenizer(sent_a, sent_b, truncation=True, max_length=self.max_len, padding="max_length")
        input_ids = torch.tensor(tokenized['input_ids'])
        token_type_ids = torch.tensor(tokenized['token_type_ids'])
        attention_mask = torch.tensor(tokenized['attention_mask'])

        # === Chuẩn bị cho tác vụ MLM ===
        labels = input_ids.clone() # Nhãn ban đầu là các token gốc.

        # Tạo ma trận xác suất để chọn token cần che (15%).
        probability_matrix = torch.full(labels.shape, 0.15)
        # Không che các token đặc biệt ([CLS], [SEP], [PAD]).
        special_tokens_mask = torch.tensor(self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True), dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # Chọn các vị trí để che dựa trên xác suất.
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Gán nhãn -100 cho các token không bị che, để hàm loss bỏ qua chúng.
        labels[~masked_indices] = -100

        # Áp dụng chiến lược 80-10-10:
        # 80% các token bị che -> thay bằng [MASK].
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% các token bị che -> thay bằng một từ ngẫu nhiên.
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        # 10% còn lại giữ nguyên token gốc.

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "masked_lm_labels": labels,
            "next_sentence_label": torch.tensor(1 if is_next else 0, dtype=torch.long)
        }


# --- Bước 2: các hàm Test có huấn luyện ---

def train_loop(model, dataloader, optimizer, epochs, device):
    """
    Hàm thực hiện vòng lặp huấn luyện.
    Nhận vào mô hình, dataloader, optimizer và huấn luyện trong `epochs` lần.
    """
    model.train() # Chuyển mô hình sang chế độ huấn luyện.
    print("Bắt đầu quá trình huấn luyện...")
    for epoch in range(epochs):
        total_loss = 0
        # Dùng tqdm để tạo thanh tiến trình.
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad() # Xóa gradient của bước trước.
            # Đưa batch dữ liệu lên device (CPU/GPU).
            inputs = {k: v.to(device) for k, v in batch.items()}
            # Forward pass: đưa dữ liệu qua mô hình.
            outputs = model(**inputs)
            loss = outputs['loss'] # Lấy loss từ đầu ra của mô hình.
            if loss is not None:
                # Backward pass: tính toán gradient.
                loss.backward()
                # Cập nhật trọng số.
                optimizer.step()
                total_loss += loss.item()
            # Cập nhật loss lên thanh tiến trình.
            progress_bar.set_postfix({'loss': loss.item() if loss else 'N/A'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} hoàn tất. Loss trung bình: {avg_loss:.4f}")
    print("Huấn luyện hoàn tất!")


def test_MLM(test_sentence="The capital of France is [MASK]."):
    """
    Kiểm thử và huấn luyện tác vụ Masked Language Model.
    Hàm này nhận một câu đầu vào để dự đoán.
    """
    print("\n" + "="*20 + " Kiểm thử Tác vụ MLM " + "="*20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng device: {device}")
    
    print(f"Kích thước corpus huấn luyện: {len(small_corpus)} cặp câu.")

    config = BertConfig(
        vocab_size=30522, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128*4
    )
    model = BertForPreTraining(config).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = PretrainingDataset(small_corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_loop(model, dataloader, optimizer, epochs=10, device=device)

    model.eval()
    print(f"\nCâu cần dự đoán: '{test_sentence}'")
    
    if '[MASK]' not in test_sentence:
        print("Lỗi: Câu đầu vào cho test_MLM phải chứa token '[MASK]'.")
        return
        
    inputs = tokenizer(test_sentence, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction_logits = outputs['prediction_logits']
    masked_index = (inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=False).item()
    
    # Lấy logits tại vị trí token bị che
    masked_token_logits = prediction_logits[0, masked_index]
    
    # Lấy top 5 dự đoán dựa trên score (logit) cao nhất
    top_5_scores, top_5_ids = torch.topk(masked_token_logits, 5)
    
    # Tính toán xác suất (độ tin cậy) tương ứng cho top 5
    top_5_probs = F.softmax(top_5_scores, dim=-1)
    
    top_5_tokens = tokenizer.convert_ids_to_tokens(top_5_ids)

    print("\nTop 5 dự đoán cho [MASK]:")
    print("-" * 50)
    print(f"{'Token':<15} | {'Score (Logit)':<15} | {'Độ tin cậy tương đối':<20}")
    print("-" * 50)
    for token, score, prob in zip(top_5_tokens, top_5_scores, top_5_probs):
        print(f"{token:<15} | {score.item():<15.4f} | {prob.item():.2%}")


def test_NSP(sent_a="The man went to the store.", sent_b="He bought a gallon of milk."):
    """
    Kiểm thử và huấn luyện tác vụ Next Sentence Prediction.
    Hàm này nhận hai câu đầu vào để dự đoán mối quan hệ.
    """
    print("\n" + "="*20 + " Kiểm thử Tác vụ NSP " + "="*20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng device: {device}")
    
    print(f"Kích thước corpus huấn luyện: {len(small_corpus)} cặp câu.")

    config = BertConfig(
        vocab_size=30522, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128*4
    )
    model = BertForPreTraining(config).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = PretrainingDataset(small_corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_loop(model, dataloader, optimizer, epochs=10, device=device)

    model.eval()
    print(f"\nKiểm tra dự đoán NSP cho cặp câu:")
    print(f"  - Câu A: '{sent_a}'")
    print(f"  - Câu B: '{sent_b}'")

    inputs = tokenizer(sent_a, sent_b, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs['seq_relationship_logits'][0] # Lấy logits cho mẫu đầu tiên
    probs = F.softmax(logits, dim=-1)
    
    # Lấy score cho từng lớp: 0 là NotNext, 1 là IsNext
    score_not_next = logits[0].item()
    score_is_next = logits[1].item()

    confidence, prediction_idx = torch.max(probs, dim=-1)
    prediction_text = "Là câu tiếp theo (IsNext)" if prediction_idx.item() == 1 else "Không phải câu tiếp theo (NotNext)"
    
    print("\n--- Phân tích của mô hình ---")
    print(f"Score cho 'Không phải câu tiếp theo': {score_not_next:.4f}")
    print(f"Score cho 'Là câu tiếp theo'        : {score_is_next:.4f}")
    print("--------------------------------")
    print(f"Dự đoán cuối cùng: '{prediction_text}' (Độ tin cậy: {confidence.item():.2%})")


if __name__ == '__main__':
    # Chạy các kiểm thử với 2 task MLM và NSP
    print("--- Chạy ví dụ cho test_MLM ---")
    test_MLM(test_sentence="BERT is a powerful [MASK].")
    
    print("\n\n--- Chạy ví dụ cho test_NSP (cặp câu liên quan) ---")
    test_NSP(sent_a="She reads a book.", sent_b="The book is about dragons.")
    
    print("\n\n--- Chạy ví dụ cho test_NSP (cặp câu không liên quan) ---")
    test_NSP(sent_a="The dog is cute.", sent_b="The sky is blue.")

    print("\n" + "="*50)
    print("Tất cả các kiểm thử có huấn luyện đã hoàn tất!")
    print("="*50)