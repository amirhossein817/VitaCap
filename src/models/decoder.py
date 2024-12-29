import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class DecoderModule(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        hidden_dim,
        num_layers,
        dropout=0.1,
        max_seq_length=50,
    ):
        super(DecoderModule, self).__init__()

        self.embed_size = embed_size
        self.max_seq_length = max_seq_length

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(
            embed_size, max_seq_length, dropout
        )

        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Initialize weights
        self._initialize_weights()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (torch.Tensor): Target sequence tokens [batch_size, seq_len].
            memory (torch.Tensor): Encoded image features [batch_size, seq_len, embed_size].
            tgt_mask (torch.Tensor, optional): Target mask [seq_len, seq_len].
            memory_mask (torch.Tensor, optional): Memory mask [seq_len, seq_len].
        Returns:
            torch.Tensor: Logits over the vocabulary [batch_size, seq_len, vocab_size].
        """
        # Embed the target sequence and add positional encoding
        tgt_embedded = self.embedding(tgt) * torch.sqrt(
            torch.tensor(self.embed_size, dtype=torch.float32)
        )
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Pass through the Transformer decoder
        decoded_output = self.transformer_decoder(
            tgt=tgt_embedded.permute(
                1, 0, 2
            ),  # Transformer expects [seq_len, batch_size, embed_size]
            memory=memory.permute(1, 0, 2),
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        # Convert to vocabulary logits
        output_logits = self.fc_out(
            decoded_output.permute(1, 0, 2)
        )  # [batch_size, seq_len, vocab_size]
        return output_logits

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_length, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / embed_size)
        )
        pe = torch.zeros(max_seq_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, embed_size]

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]  # Add positional encoding to input
        return self.dropout(x)


# Example usage
if __name__ == "__main__":
    vocab_size = 5000
    embed_size = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 6
    max_seq_length = 50

    decoder = DecoderModule(
        vocab_size,
        embed_size,
        num_heads,
        hidden_dim,
        num_layers,
        max_seq_length=max_seq_length,
    )
    tgt = torch.randint(0, vocab_size, (32, max_seq_length))  # Dummy target tokens
    memory = torch.randn(32, 20, embed_size)  # Dummy encoded image features

    output = decoder(tgt, memory)
    print("Output shape:", output.shape)  # [batch_size, seq_len, vocab_size]
