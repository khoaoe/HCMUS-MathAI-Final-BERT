"""
Triển khai mô hình BERT (Bidirectional Encoder Representations from Transformers)
Dựa trên bài báo: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
của Devlin et al., 2018

File này implement đầy đủ kiến trúc BERT bao gồm:
- Multi-Head Self-Attention 
- Transformer Encoder
- Masked Language Model (MLM)
- Next Sentence Prediction (NSP)
"""
# ========================= Phần Import Thư viện =========================

# --- PyTorch Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
# torch: Framework deep learning chính
# nn: Module chứa các layers và loss functions  
# F: Functional API cho các phép toán không có trạng thái
# AdamW: Optimizer với weight decay tách biệt

# --- Thư viện Toán học và Tiện ích Python ---
import math  # Dùng cho sqrt trong scaled attention 
import random  # Tạo ngẫu nhiên cho MLM và NSP
from typing import Dict, List, Tuple, Optional, Union  # Type hints
import time  
from dataclasses import dataclass  # Decorator cho config class

# --- Thư viện Hugging Face ---
from transformers import BertTokenizer
# BertTokenizer: Implement WordPiece tokenization
# Vocab size ~30K tokens, xử lý OOV bằng subword units

# --- PyTorch Data Utils ---
from torch.utils.data import Dataset, DataLoader
# Dataset: Abstract class cho custom datasets
# DataLoader: Batching, shuffling, multiprocessing

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns  
from tqdm import tqdm  # Progress bars cho training loops

# ========================= Cấu hình (Configuration) =========================
@dataclass
class BertConfig:
    """
    Cấu hình cho mô hình BERT.
    
    Các hyperparameters chính theo notation trong paper:
    - L (num_hidden_layers): Số lớp Transformer 
    - H (hidden_size): Hidden dimension
    - A (num_attention_heads): Số attention heads
    - Feed-forward size = 4*H (theo paper gốc)
    
    BERT-Base: L=12, H=768, A=12 (110M params)
    BERT-Large: L=24, H=1024, A=16 (340M params)
    """
    vocab_size: int = 30522                     # Kích thước từ vựng WordPiece
    hidden_size: int = 768                      # Hidden size H 
    num_hidden_layers: int = 12                 # Số layers L
    num_attention_heads: int = 12               # Số heads A
    intermediate_size: int = 3072               # FFN size = 4*H
    hidden_dropout_prob: float = 0.1            # Dropout cho hidden states
    attention_probs_dropout_prob: float = 0.1   # Dropout cho attention weights
    max_position_embeddings: int = 512          # Max sequence length
    type_vocab_size: int = 2                    # Segment types (A/B)
    initializer_range: float = 0.02             # Std cho normal initialization


# ========================= Các Module con của BERT =========================

class BertEmbeddings(nn.Module):
    """
    Tầng Embedding của BERT.
    
    BERT sử dụng tổng của 3 embeddings:
    1. Token Embeddings: Biểu diễn từng WordPiece token
    2. Segment Embeddings: Phân biệt câu A và B (cho NSP task)  
    3. Position Embeddings: Thông tin vị trí (learned, không phải sinusoidal)
    
    Công thức: E = E_token + E_segment + E_position
    Sau đó qua LayerNorm và Dropout.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        # Token embeddings với padding_idx=0 ([PAD] token)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=0
        )
        
        # Position embeddings - BERT học position thay vì dùng sinusoidal
        # Khác với Transformer gốc (Vaswani et al., 2017)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # Segment embeddings để phân biệt câu A/B
        # Cần thiết cho NSP task
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, 
            config.hidden_size
        )
        
        # LayerNorm (Ba et al., 2016) với epsilon nhỏ tránh chia 0
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        # Dropout để regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, 
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass của embedding layer.
        
        Args:
            input_ids: Token IDs shape (batch_size, seq_length)
            token_type_ids: Segment IDs shape (batch_size, seq_length)
            
        Returns:
            embeddings: shape (batch_size, seq_length, hidden_size)
        """
        seq_length = input_ids.size(1)
        
        # Tạo position IDs từ 0 đến seq_length-1
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Default segment 0 nếu không có token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Lấy 3 loại embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Cộng tổng 3 embeddings (Figure 2 trong paper)
        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        # LayerNorm và Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    
    Công thức Scaled Dot-Product Attention:
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    Multi-Head cho phép mô hình jointly attend từ nhiều representation 
    subspaces khác nhau:
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    với head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    BERT sử dụng h=12 heads cho Base, h=16 cho Large.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        # Kiểm tra hidden_size chia hết cho num_heads
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections cho Q, K, V
        # W^Q, W^K, W^V trong paper
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout cho attention probabilities
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor từ (batch, seq_len, hidden) sang (batch, heads, seq_len, head_size)
        để tính attention cho từng head độc lập.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass của Multi-Head Attention.
        
        Args:
            hidden_states: shape (batch, seq_len, hidden_size)
            attention_mask: shape (batch, 1, 1, seq_len), giá trị -10000 cho masked positions
            
        Returns:
            context_layer: shape (batch, seq_len, hidden_size)
            attention_probs: shape (batch, heads, seq_len, seq_len) để visualization
        """
        # Tính Q, K, V projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Reshape cho multi-head: (batch, heads, seq_len, head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Tính attention scores: QK^T
        # shape: (batch, heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # Scale bởi √d_k để tránh softmax saturation khi d_k lớn
        # Đây là "Scaled" trong "Scaled Dot-Product Attention"
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Áp dụng mask: -10000 cho positions cần mask
            # Sau softmax sẽ thành ~0
            attention_scores = attention_scores + attention_mask

        # Normalize scores thành probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # Dropout cho regularization
        attention_probs = self.dropout(attention_probs)

        # Tính weighted sum: attention_probs @ V
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape về (batch, seq_len, all_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """
    Output layer sau Multi-Head Attention.
    
    Bao gồm:
    1. Linear projection 
    2. Dropout
    3. Residual connection (Add)
    4. Layer Normalization (Norm)
    
    Đây là pattern "Add & Norm" trong Transformer.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Output từ attention
            input_tensor: Input của attention (cho residual connection)
        """
        # Linear projection
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Residual connection + LayerNorm
        # Giúp training deep networks (He et al., 2016)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    Complete Attention block = Multi-Head Attention + Add & Norm.
    Đây là một sub-layer trong Transformer encoder.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = MultiHeadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-head attention
        self_outputs = self.self(hidden_states, attention_mask)
        # Add & Norm
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # Thêm attention probs nếu cần
        return outputs


class BertIntermediate(nn.Module):
    """
    Feed-Forward Network (FFN) đầu tiên trong Transformer block.
    
    Expand dimension từ H -> 4H (hoặc intermediate_size).
    Sử dụng GELU activation thay vì ReLU.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # GELU: Gaussian Error Linear Unit - smoother than ReLU
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    FFN thứ hai: thu hẹp dimension từ 4H -> H.
    Cũng có Add & Norm pattern.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Output từ BertIntermediate
            input_tensor: Input của FFN (từ attention output)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Residual + LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    Một Transformer Encoder Layer hoàn chỉnh.
    
    Cấu trúc:
    1. Multi-Head Self-Attention + Add & Norm
    2. Position-wise FFN + Add & Norm
    
    BERT-Base có L=12 layers, BERT-Large có L=24 layers.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass qua 1 Transformer layer.
        
        Args:
            hidden_states: shape (batch, seq_len, hidden_size)
            attention_mask: shape (batch, 1, 1, seq_len)
            
        Returns:
            layer_output: shape (batch, seq_len, hidden_size)
            attention_probs: shape (batch, heads, seq_len, seq_len)
        """
        # Self-Attention sub-layer
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]

        # Feed-Forward sub-layer
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    """
    Stack của L Transformer Encoder layers.
    
    BERT-Base: 12 layers
    BERT-Large: 24 layers
    
    Mỗi layer xử lý độc lập nhưng tuần tự.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        # Tạo L layers
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                output_attentions: bool = False) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        Forward qua tất cả L layers.
        
        Returns:
            Dict với:
            - 'last_hidden_state': Output của layer cuối cùng
            - 'attentions': Tuple của attention weights mỗi layer (nếu output_attentions=True)
        """
        all_attentions = () if output_attentions else None
        
        # Forward qua từng layer tuần tự
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]  # Update hidden states
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        return {
            'last_hidden_state': hidden_states,
            'attentions': all_attentions
        }


class BertPooler(nn.Module):
    """
    Pooling layer để tạo sentence representation.
    
    Lấy hidden state của [CLS] token (vị trí 0) và transform qua:
    1. Linear layer  
    2. Tanh activation
    
    Output này dùng cho NSP task và sentence-level tasks.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: shape (batch, seq_len, hidden_size)
            
        Returns:
            pooled_output: shape (batch, hidden_size)
        """
        # Lấy [CLS] token (position 0)
        first_token_tensor = hidden_states[:, 0]
        # Transform
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """
    Transform layer trước khi predict masked tokens.
    
    Architecture:
    1. Linear (hidden_size -> hidden_size)
    2. GELU activation
    3. LayerNorm
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
    Prediction head cho Masked Language Model (MLM) task.
    
    Architecture:
    1. Transform layer
    2. Decoder (project về vocab size)
    
    Decoder weights được tied với input embeddings để giảm parameters
    """
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        
        # Decoder với tied weights
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights  # Tie weights
        
        # Bias riêng cho decoder
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: shape (batch, seq_len, hidden_size)
            
        Returns:
            prediction_scores: shape (batch, seq_len, vocab_size)
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# ========================= Mô hình BERT Hoàn chỉnh =========================

class BertModel(nn.Module):
    """
    BERT model cơ bản (không có task heads).
    
    Architecture:
    1. Embeddings (token + segment + position)
    2. Encoder (L Transformer layers)  
    3. Pooler (cho [CLS] representation)
    
    Đây là base model, cần thêm task-specific heads cho downstream tasks.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids: torch.Tensor, 
                token_type_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass của BERT base model.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            token_type_ids: Segment IDs, shape (batch, seq_len)
            attention_mask: Mask với 1 cho real tokens, 0 cho padding
            output_attentions: Có return attention weights không
            
        Returns:
            Dict với:
            - 'last_hidden_state': shape (batch, seq_len, hidden_size)
            - 'pooler_output': shape (batch, hidden_size) 
            - 'attentions': Tuple của attention weights nếu requested
        """
        # Default masks nếu không provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Tạo extended attention mask cho broadcasting
        # Từ (batch, seq_len) -> (batch, 1, 1, seq_len)
        # Shape này phù hợp với (batch, heads, seq_len, seq_len) của attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        
        # Chuyển mask sang additive mask
        # 0 -> -10000 (bị masked), 1 -> 0 (không masked)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Forward qua các components
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, output_attentions)
        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output)
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'attentions': encoder_outputs['attentions']
        }


class BertForPreTraining(nn.Module):
    """
    BERT model với pre-training heads.
    
    Gồm 2 pre-training tasks:
    1. Masked Language Model (MLM): Predict masked tokens
    2. Next Sentence Prediction (NSP): Binary classification
    
    Loss = Loss_MLM + Loss_NSP
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        
        # MLM head với tied embeddings
        self.cls_mlm = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        
        # NSP head: binary classifier
        self.cls_nsp = nn.Linear(config.hidden_size, 2)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights theo paper:
        - Normal distribution với std = initializer_range (0.02)
        - LayerNorm: bias=0, weight=1
        - Linear bias: 0
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Normal initialization
                module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: bias=0, weight=1
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # Linear bias=0
                module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids, attention_mask, 
                masked_lm_labels=None, next_sentence_label=None, 
                output_attentions=False):
        """
        Forward pass cho pre-training.
        
        Args:
            input_ids: Token IDs với masked positions
            token_type_ids: Segment IDs (0 cho câu A, 1 cho câu B)
            attention_mask: Attention mask
            masked_lm_labels: Labels cho MLM, -100 cho non-masked
            next_sentence_label: 0 (NotNext) hoặc 1 (IsNext)
            output_attentions: Return attention weights
            
        Returns:
            Dict với:
            - 'loss': Combined loss nếu có labels
            - 'prediction_logits': MLM predictions shape (batch, seq_len, vocab_size)
            - 'seq_relationship_logits': NSP predictions shape (batch, 2)
            - 'attentions': Attention weights nếu requested
        """
        # Forward qua base BERT
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_attentions)
        sequence_output = outputs['last_hidden_state']  # For MLM
        pooled_output = outputs['pooler_output']  # For NSP

        # MLM predictions cho mọi positions
        prediction_scores = self.cls_mlm(sequence_output)
        
        # NSP prediction từ [CLS]
        seq_relationship_score = self.cls_nsp(pooled_output)

        total_loss = None
        if masked_lm_labels is not None and next_sentence_label is not None:
            # Cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            
            # MLM loss: chỉ tính cho masked positions (ignore -100)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size), 
                masked_lm_labels.view(-1)
            )
            
            # NSP loss: binary classification
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), 
                next_sentence_label.view(-1)
            )
            
            # Combined loss (Equation 3.4 trong paper)
            total_loss = masked_lm_loss + next_sentence_loss

        return {
            'loss': total_loss,
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score,
            'attentions': outputs['attentions']
        }


# ========================= Phần Huấn luyện và Kiểm thử =========================

# --- Corpus nhỏ để demo ---
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
    Dataset cho BERT pre-training với MLM và NSP tasks.
    
    MLM Strategy:
    - Mask 15% tokens ngẫu nhiên
    - 80% -> [MASK]
    - 10% -> random token
    - 10% -> giữ nguyên
    
    NSP Strategy:
    - 50% câu B thực sự theo sau câu A (IsNext)
    - 50% câu B random (NotNext)
    """
    def __init__(self, corpus, tokenizer, max_len=64):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        """
        Tạo một training example với MLM và NSP.
        
        Returns:
            Dict với:
            - input_ids: Token IDs với masked positions
            - token_type_ids: Segment IDs
            - attention_mask: 1 cho real tokens, 0 cho padding
            - masked_lm_labels: Original token IDs cho masked positions, -100 cho others
            - next_sentence_label: 1 (IsNext) hoặc 0 (NotNext)
        """
        # === Chuẩn bị cho NSP ===
        sent_a, sent_b = self.corpus[idx]
        is_next = True
        
        # 50% xác suất chọn random sentence
        if random.random() < 0.5:
            rand_idx = random.randint(0, len(self.corpus) - 1)
            sent_b = self.corpus[rand_idx][1]
            is_next = False

        # Tokenize với format [CLS] A [SEP] B [SEP]
        tokenized = self.tokenizer(
            sent_a, sent_b, 
            truncation=True, 
            max_length=self.max_len, 
            padding="max_length"
        )
        
        input_ids = torch.tensor(tokenized['input_ids'])
        token_type_ids = torch.tensor(tokenized['token_type_ids'])
        attention_mask = torch.tensor(tokenized['attention_mask'])

        # === Chuẩn bị cho MLM ===
        labels = input_ids.clone()
        
        # Tạo probability matrix 15% cho masking
        probability_matrix = torch.full(labels.shape, 0.15)
        
        # Không mask special tokens ([CLS], [SEP], [PAD])
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True), 
            dtype=torch.bool
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Chọn positions để mask dựa trên xác suất
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set label -100 cho non-masked positions (ignored trong loss)
        labels[~masked_indices] = -100

        # Áp dụng chiến lược 80-10-10
        # 80% masked positions -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% masked positions -> random token  
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% còn lại giữ nguyên

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "masked_lm_labels": labels,
            "next_sentence_label": torch.tensor(1 if is_next else 0, dtype=torch.long)
        }


def train_loop(model, dataloader, optimizer, epochs, device):
    """
    Training loop cho BERT pre-training.
    
    Args:
        model: BertForPreTraining model
        dataloader: DataLoader với PretrainingDataset
        optimizer: AdamW optimizer
        epochs: Số epochs
        device: 'cuda' hoặc 'cpu'
    """
    model.train()
    print("Bắt đầu quá trình huấn luyện BERT...")
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch lên device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs['loss']
            
            if loss is not None:
                # Backward pass
                loss.backward()
                
                # Gradient clipping (optional nhưng recommended)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                
                total_loss += loss.item()
                
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() if loss else 'N/A'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} hoàn tất. Loss trung bình: {avg_loss:.4f}")
        
    print("Huấn luyện hoàn tất!")


def test_MLM(test_sentence="The capital of France is [MASK]."):
    """
    Demo Masked Language Model prediction.
    
    Huấn luyện BERT nhỏ và test khả năng predict masked tokens.
    """
    print("\n" + "="*20 + " Kiểm thử Tác vụ MLM " + "="*20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng device: {device}")
    
    # Config nhỏ cho demo (thực tế dùng hidden_size=768, layers=12)
    config = BertConfig(
        vocab_size=30522, 
        hidden_size=128,      # Giảm từ 768
        num_hidden_layers=2,  # Giảm từ 12
        num_attention_heads=2,  # Giảm từ 12
        intermediate_size=128*4
    )
    
    model = BertForPreTraining(config).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tạo dataset và dataloader
    dataset = PretrainingDataset(small_corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer với learning rate nhỏ
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train vài epochs
    train_loop(model, dataloader, optimizer, epochs=10, device=device)

    # Test prediction
    model.eval()
    print(f"\nCâu cần dự đoán: '{test_sentence}'")
    
    if '[MASK]' not in test_sentence:
        print("Lỗi: Câu phải chứa token '[MASK]'.")
        return
        
    inputs = tokenizer(test_sentence, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction_logits = outputs['prediction_logits']
    
    # Tìm vị trí [MASK]
    masked_index = (inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=False).item()
    
    # Lấy logits tại vị trí masked
    masked_token_logits = prediction_logits[0, masked_index]
    
    # Top 5 predictions
    top_5_scores, top_5_ids = torch.topk(masked_token_logits, 5)
    top_5_probs = F.softmax(top_5_scores, dim=-1)
    top_5_tokens = tokenizer.convert_ids_to_tokens(top_5_ids)

    print("\nTop 5 dự đoán cho [MASK]:")
    print("-" * 50)
    print(f"{'Token':<15} | {'Score (Logit)':<15} | {'Xác suất':<20}")
    print("-" * 50)
    for token, score, prob in zip(top_5_tokens, top_5_scores, top_5_probs):
        print(f"{token:<15} | {score.item():<15.4f} | {prob.item():.2%}")


def test_NSP(sent_a="The man went to the store.", sent_b="He bought a gallon of milk."):
    """
    Demo Next Sentence Prediction.
    
    Test khả năng phân biệt câu liên tiếp vs random.
    """
    print("\n" + "="*20 + " Kiểm thử Tác vụ NSP " + "="*20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Config và setup tương tự MLM test
    config = BertConfig(
        vocab_size=30522, 
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128*4
    )
    
    model = BertForPreTraining(config).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = PretrainingDataset(small_corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train
    train_loop(model, dataloader, optimizer, epochs=10, device=device)

    # Test NSP
    model.eval()
    print(f"\nKiểm tra NSP cho cặp câu:")
    print(f"  - Câu A: '{sent_a}'")
    print(f"  - Câu B: '{sent_b}'")

    inputs = tokenizer(sent_a, sent_b, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs['seq_relationship_logits'][0]
    probs = F.softmax(logits, dim=-1)
    
    # Scores cho 2 classes
    score_not_next = logits[0].item()
    score_is_next = logits[1].item()

    confidence, prediction_idx = torch.max(probs, dim=-1)
    prediction_text = "IsNext" if prediction_idx.item() == 1 else "NotNext"
    
    print("\n--- Phân tích NSP ---")
    print(f"Score 'NotNext': {score_not_next:.4f}")
    print(f"Score 'IsNext': {score_is_next:.4f}")
    print("--------------------")
    print(f"Dự đoán: '{prediction_text}' (Độ tin cậy: {confidence.item():.2%})")


if __name__ == '__main__':
    """
    Main entry point - chạy demo cho cả MLM và NSP.
    """
    print("=== BERT Implementation Demo ===")
    
    # Test MLM
    print("\n--- Demo 1: Masked Language Model ---")
    test_MLM(test_sentence="BERT is a powerful [MASK].")
    
    # Test NSP
    print("\n\n--- Demo 2: Next Sentence Prediction (câu liên quan) ---")
    test_NSP(sent_a="She reads a book.", sent_b="The book is about dragons.")
    
    print("\n\n--- Demo 3: Next Sentence Prediction (câu không liên quan) ---")
    test_NSP(sent_a="The dog is cute.", sent_b="The sky is blue.")

    print("\n" + "="*50)
    print("Demo hoàn tất!")
    print("="*50)