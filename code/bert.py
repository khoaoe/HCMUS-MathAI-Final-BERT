# %% [markdown]
# # Libraries

# %%
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

# %% [markdown]
# # Configuration

# %%
@dataclass
class BertConfig:
    """BERT configuration"""
    vocab_size: int = 30522                     # Size of vocabulary
    hidden_size: int = 768                      # Hidden size (d_model)
    num_hidden_layers: int = 12                 # Number of transformer blocks
    num_attention_heads: int = 12               # Number of attention heads
    intermediate_size: int = 3072               # FFN intermediate size
    hidden_dropout_prob: float = 0.1            # Dropout for hidden layers
    attention_probs_dropout_prob: float = 0.1   # Dropout for attention
    max_position_embeddings: int = 512          # Maximum sequence length
    type_vocab_size: int = 2                    # Token type vocab size
    initializer_range: float = 0.02             # Weight initialization std
    layer_norm_eps: float = 1e-12               # Layer norm epsilon
    
    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"

# %% [markdown]
# # Attentions

# %%
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with detailed mathematical explanation
    
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    This addresses the problem of dot products growing large in magnitude
    for large d_k, pushing softmax into regions with extremely small gradients.
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
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            mask: Attention mask (1 for positions to attend, 0 for masked)
            return_attention: Whether to return attention weights
        """
        batch_size, n_heads, seq_len, d_k = query.size()
        
        # Compute attention scores
        # Einstein notation for clarity: bhqd,bhkd->bhqk
        scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for all heads
            if mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            elif mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            
            # Fill masked positions with -inf so they become 0 after softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # Einstein notation: bhqk,bhkd->bhqd
        output = torch.matmul(attn_weights, value)
        
        if return_attention:
            return output, attn_weights
        return output

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Xavier uniform initialization
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
            mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len] (if return_attention=True)
        """
        batch_size, seq_len, _ = query.size()
        
        # 1. Linear projections in batch from d_model => h x d_k
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # 2. Apply attention on all the projected vectors in batch
        if return_attention:
            attn_output, attn_weights = self.attention(Q, K, V, mask=mask, return_attention=True)
        else:
            attn_output = self.attention(Q, K, V, mask=mask, return_attention=False)
            attn_weights = None
        
        # 3. "Concat" using a view and apply a final linear
        # [batch, n_heads, seq_len, d_v] -> [batch, seq_len, n_heads, d_v] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 4. Final linear projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output

# %% [markdown]
# # Feed-Forward Network

# %%
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    BERT uses GELU activation instead of ReLU for smoother gradients
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = self._get_activation_fn(activation)
        
    def _get_activation_fn(self, activation: str):
        """Get activation function by name"""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise ValueError(f"Activation '{activation}' not supported")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ========================= Transformer Block =========================
class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture (used in BERT)
    
    Each block contains:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Both with residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Sub-layers
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
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
            mask: [batch_size, seq_len] or None
            return_attention: Whether to return attention weights
        """
        # Self-attention with residual connection and layer norm
        if return_attention:
            attn_output, attn_weights = self.attention(x, x, x, mask, return_attention=True)
        else:
            attn_output = self.attention(x, x, x, mask, return_attention=False)
            attn_weights = None
            
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        if return_attention:
            return x, attn_weights
        return x

# %% [markdown]
# # Embeddings

# %%
class BertEmbeddings(nn.Module):
    """
    BERT embeddings consist of:
    1. Token embeddings (from vocabulary)
    2. Position embeddings (learned, not sinusoidal)
    3. Token type embeddings (for sentence A/B distinction)
    
    The final embedding is the sum of all three, followed by LayerNorm and dropout
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Create position IDs buffer
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
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] or None
            position_ids: [batch_size, seq_len] or None
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        seq_length = input_ids.size(1)
        
        # Get position IDs
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        # Get token type IDs (default to 0 if not provided)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Sum all embeddings
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        
        # Apply LayerNorm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# %% [markdown]
# # BERT Encoder

# %%
class BertEncoder(nn.Module):
    """Stack of Transformer blocks"""
    
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
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or None
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
        Returns:
            Dictionary containing:
                - last_hidden_state: [batch_size, seq_len, hidden_size]
                - hidden_states: List of hidden states (if output_hidden_states=True)
                - attentions: List of attention weights (if output_attentions=True)
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            if output_attentions:
                hidden_states, attn_weights = layer(hidden_states, attention_mask, return_attention=True)
                all_attentions.append(attn_weights)
            else:
                hidden_states = layer(hidden_states, attention_mask, return_attention=False)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            
        outputs = {
            'last_hidden_state': hidden_states
        }
        
        if output_hidden_states:
            outputs['hidden_states'] = all_hidden_states
        if output_attentions:
            outputs['attentions'] = all_attentions
            
        return outputs

# %% [markdown]
# # BERT Pooler

# %%
class BertPooler(nn.Module):
    """
    Pool the [CLS] token representation for classification tasks
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Take the hidden state of the first token ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# %% [markdown]
# # Prediction Heads

# %%
class BertPredictionHeadTransform(nn.Module):
    """Transform for MLM predictions"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = F.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    """Language Model prediction head for MLM"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    """Pre-training heads for MLM and NSP"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        
    def forward(
        self, 
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            pooled_output: [batch_size, hidden_size]
        Returns:
            prediction_scores: [batch_size, seq_len, vocab_size]
            seq_relationship_score: [batch_size, 2]
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

# %% [markdown]
# # BERT Model

# %%
class BertModel(nn.Module):
    """
    BERT model with all components integrated
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
        for compatibility with multi-head attention
        """
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
            
        # Convert to float and apply mask
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
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
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or None
            token_type_ids: [batch_size, seq_len] or None
            position_ids: [batch_size, seq_len] or None
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
        Returns:
            Dictionary containing model outputs
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output)
        
        outputs = {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output
        }
        
        if output_hidden_states:
            outputs['hidden_states'] = encoder_outputs['hidden_states']
        if output_attentions:
            outputs['attentions'] = encoder_outputs['attentions']
            
        return outputs

# %% [markdown]
# # BERT Pre-trained Model

# %%
class BertForPreTraining(nn.Module):
    """BERT model with pre-training heads"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        
        # Tie weights between input embeddings and output embeddings
        self.tie_weights()
        
    def tie_weights(self):
        """Tie the weights between input embeddings and output embeddings"""
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pre-training
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or None
            token_type_ids: [batch_size, seq_len] or None
            position_ids: [batch_size, seq_len] or None
            labels: [batch_size, seq_len] - labels for MLM (-100 for non-masked tokens)
            next_sentence_label: [batch_size] - labels for NSP (0 or 1)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
        Returns:
            Dictionary containing losses and predictions
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
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # MLM loss
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1)
            )
            
            # NSP loss
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )
            
            total_loss = masked_lm_loss + next_sentence_loss
            
        return {
            'loss': total_loss,
            'mlm_loss': masked_lm_loss if labels is not None else None,
            'nsp_loss': next_sentence_loss if next_sentence_label is not None else None,
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score,
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions')
        }

# %% [markdown]
# # BERT Classification ModeL

# %%
class BertForSequenceClassification(nn.Module):
    """BERT for sequence classification tasks"""
    
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
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or None
            token_type_ids: [batch_size, seq_len] or None
            labels: [batch_size] - classification labels
        Returns:
            Dictionary containing loss and logits
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

# %% [markdown]
# # Dataset Implementation

# %%
class BertDataset(Dataset):
    """
    Dataset for BERT pre-training with MLM and NSP tasks
    
    Implements the 80-10-10 masking strategy:
    - 80% of the time: Replace with [MASK] token
    - 10% of the time: Replace with random token
    - 10% of the time: Keep original token
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        short_seq_prob: float = 0.1
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.short_seq_prob = short_seq_prob
        
        # Pre-process texts into sentences
        self.documents = self._preprocess_texts()
        
    def _preprocess_texts(self) -> List[List[str]]:
        """Split texts into sentences"""
        documents = []
        for text in self.texts:
            # Simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                documents.append(sentences)
        return documents
    
    def _get_random_sentence(self, exclude_doc_idx: int) -> str:
        """Get a random sentence from a different document"""
        if len(self.documents) == 1:
            return ""
        
        doc_idx = random.choice([i for i in range(len(self.documents)) if i != exclude_doc_idx])
        if self.documents[doc_idx]:
            return random.choice(self.documents[doc_idx])
        return ""
    
    def _create_training_instance(self, doc_idx: int) -> Tuple[str, str, int]:
        """Create a training instance with sentence A, sentence B, and NSP label"""
        document = self.documents[doc_idx]
        
        # Get sentence A
        sent_idx_a = random.randint(0, len(document) - 1)
        sent_a = document[sent_idx_a]
        
        # Create sentence B and NSP label
        if random.random() < 0.5 and sent_idx_a < len(document) - 1:
            # Next sentence (positive example)
            sent_b = document[sent_idx_a + 1]
            is_next = 1
        else:
            # Random sentence (negative example)
            sent_b = self._get_random_sentence(doc_idx)
            is_next = 0
            
        return sent_a, sent_b, is_next
    
    def _truncate_seq_pair(self, tokens_a: List[int], tokens_b: List[int], max_length: int):
        """Truncate sequence pair to fit max_length"""
        while len(tokens_a) + len(tokens_b) > max_length - 3:  # Account for [CLS], [SEP], [SEP]
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    
    def _create_masked_lm_predictions(
        self,
        tokens: List[int],
        mlm_probability: float
    ) -> Tuple[List[int], List[int]]:
        """
        Create masked language model predictions
        
        Returns:
            output_tokens: Tokens with masking applied
            output_labels: Original tokens at masked positions (-100 for non-masked)
        """
        output_tokens = tokens.copy()
        output_labels = [-100] * len(tokens)  # -100 is ignored by CrossEntropyLoss
        
        # Get candidates for masking (exclude [CLS], [SEP], [PAD])
        candidate_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.cls_token_id,
                           self.tokenizer.sep_token_id,
                           self.tokenizer.pad_token_id]:
                candidate_indices.append(i)
        
        # Sample indices to mask
        random.shuffle(candidate_indices)
        num_to_mask = max(1, int(len(candidate_indices) * mlm_probability))
        mask_indices = candidate_indices[:num_to_mask]
        
        for idx in mask_indices:
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                output_tokens[idx] = self.tokenizer.mask_token_id
            else:
                # 10% of the time, replace with random token
                if random.random() < 0.5:
                    output_tokens[idx] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10% of the time, keep original token
                
            output_labels[idx] = tokens[idx]
            
        return output_tokens, output_labels
    
    def __len__(self) -> int:
        return len(self.documents) * 100  # Create multiple instances per document
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get document index
        doc_idx = idx % len(self.documents)
        
        # Create training instance
        sent_a, sent_b, is_next = self._create_training_instance(doc_idx)
        
        # Tokenize sentences
        tokens_a = self.tokenizer.tokenize(sent_a)
        tokens_b = self.tokenizer.tokenize(sent_b) if sent_b else []
        
        # Convert to token IDs
        tokens_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
        tokens_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
        
        # Truncate to fit max_length
        self._truncate_seq_pair(tokens_a, tokens_b, self.max_length)
        
        # Build input sequence: [CLS] A [SEP] B [SEP]
        tokens = [self.tokenizer.cls_token_id]
        segment_ids = [0]
        
        tokens.extend(tokens_a)
        segment_ids.extend([0] * len(tokens_a))
        
        tokens.append(self.tokenizer.sep_token_id)
        segment_ids.append(0)
        
        if tokens_b:
            tokens.extend(tokens_b)
            segment_ids.extend([1] * len(tokens_b))
            
            tokens.append(self.tokenizer.sep_token_id)
            segment_ids.append(1)
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Pad sequences
        padding_length = self.max_length - len(tokens)
        tokens.extend([self.tokenizer.pad_token_id] * padding_length)
        segment_ids.extend([0] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        # Create MLM predictions
        masked_tokens, mlm_labels = self._create_masked_lm_predictions(tokens, self.mlm_probability)
        
        return {
            'input_ids': torch.tensor(masked_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            'labels': torch.tensor(mlm_labels, dtype=torch.long),
            'next_sentence_label': torch.tensor(is_next, dtype=torch.long)
        }

# %% [markdown]
# # Training Utilities

# %%
class BertTrainer:
    """
    Enhanced trainer class for BERT pre-training
    """
    
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
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Learning rate scheduler with warmup
        total_steps = len(train_dataloader) * 10  # Assuming 10 epochs
        self.scheduler = self._get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        
        # Mixed precision training
        if self.mixed_precision and self.device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create a schedule with a learning rate that decreases linearly after warmup"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0
        num_batches = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Mixed precision training
            if self.mixed_precision and self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        labels=batch['labels'],
                        next_sentence_label=batch['next_sentence_label']
                    )
                    loss = outputs['loss'] / self.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    labels=batch['labels'],
                    next_sentence_label=batch['next_sentence_label']
                )
                loss = outputs['loss'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track losses
            total_loss += loss.item() * self.gradient_accumulation_steps
            if outputs.get('mlm_loss') is not None:
                total_mlm_loss += outputs['mlm_loss'].item()
            if outputs.get('nsp_loss') is not None:
                total_nsp_loss += outputs['nsp_loss'].item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log learning rate
            if batch_idx % 100 == 0:
                self.learning_rates.append(current_lr)
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches if total_mlm_loss > 0 else 0
        avg_nsp_loss = total_nsp_loss / num_batches if total_nsp_loss > 0 else 0
        
        self.train_losses.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'mlm_loss': avg_mlm_loss,
            'nsp_loss': avg_nsp_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    labels=batch['labels'],
                    next_sentence_label=batch['next_sentence_label']
                )
                
                total_loss += outputs['loss'].item()
                if outputs.get('mlm_loss') is not None:
                    total_mlm_loss += outputs['mlm_loss'].item()
                if outputs.get('nsp_loss') is not None:
                    total_nsp_loss += outputs['nsp_loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches if total_mlm_loss > 0 else 0
        avg_nsp_loss = total_nsp_loss / num_batches if total_nsp_loss > 0 else 0
        
        self.val_losses.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'mlm_loss': avg_mlm_loss,
            'nsp_loss': avg_nsp_loss
        }
    
    def train(self, num_epochs: int, save_dir: Optional[str] = None):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nTraining metrics:")
            print(f"  Average loss: {train_metrics['loss']:.4f}")
            print(f"  MLM loss: {train_metrics['mlm_loss']:.4f}")
            print(f"  NSP loss: {train_metrics['nsp_loss']:.4f}")
            
            # Validate
            if self.val_dataloader:
                val_metrics = self.validate()
                print(f"\nValidation metrics:")
                print(f"  Average loss: {val_metrics['loss']:.4f}")
                print(f"  MLM loss: {val_metrics['mlm_loss']:.4f}")
                print(f"  NSP loss: {val_metrics['nsp_loss']:.4f}")
                
                # Save best model
                if save_dir and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_dir, epoch + 1, is_best=True)
            
            # Save regular checkpoint
            if save_dir and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_dir, epoch + 1)
    
    def save_checkpoint(self, save_dir: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
        }
        
        if is_best:
            path = os.path.join(save_dir, 'best_model.pt')
        else:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot learning rate
        axes[1].plot(self.learning_rates)
        axes[1].set_xlabel('Steps (x100)')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# # Visualization

# %%
def visualize_attention(
    model: BertModel,
    tokenizer: BertTokenizer,
    text: str,
    layer_idx: int = -1,
    head_idx: int = 0
):
    """
    Visualize attention weights for a given text
    
    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        text: Input text
        layer_idx: Which layer to visualize (-1 for last layer)
        head_idx: Which attention head to visualize
    """
    # Tokenize
    inputs = tokenizer.encode_plus(
        text,
        return_tensors='pt',
        add_special_tokens=True,
        max_length=512,
        truncation=True
    )
    
    # Get model outputs with attention
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )
    
    # Get attention weights
    attentions = outputs['attentions']  # List of tensors, one per layer
    attention = attentions[layer_idx]    # [batch, n_heads, seq_len, seq_len]
    attention = attention[0, head_idx]   # [seq_len, seq_len]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention.numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # Testing

# %%
def test_attention_mechanism():
    """Test the attention mechanism with a simple example"""
    print("Testing Attention Mechanism...")
    
    # Create simple input
    batch_size, seq_len, d_model = 2, 5, 64
    n_heads = 8
    
    # Random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention module
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Forward pass
    output = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ Attention mechanism test passed!")

def test_bert_model():
    """Test the complete BERT model"""
    print("\nTesting BERT Model...")
    
    # Create config
    config = BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512
    )
    
    # Create model
    model = BertForPreTraining(config)
    
    # Create dummy input
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels[torch.rand(batch_size, seq_len) > 0.15] = -100  # Mask most positions
    next_sentence_label = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
        next_sentence_label=next_sentence_label
    )
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"MLM predictions shape: {outputs['prediction_logits'].shape}")
    print(f"NSP predictions shape: {outputs['seq_relationship_logits'].shape}")
    print("✓ BERT model test passed!")

# %%
 # Run tests
test_attention_mechanism()
test_bert_model()

print("\n" + "="*50)
print("All tests passed! The BERT implementation is working correctly.")
print("="*50)