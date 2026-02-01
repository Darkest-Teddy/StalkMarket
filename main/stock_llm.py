import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class StockLLM(nn.Module):
    def __init__(self, hidden=32, n_layer=2, n_head=2):
        super().__init__()
        config = GPT2Config(
            n_embd=hidden,
            n_layer=n_layer,
            n_head=n_head,
            vocab_size=1_000,   # dummy
            n_positions=128,
            n_ctx=128,
        )
        self.backbone = GPT2Model(config)
        self.input_proj = nn.Linear(1, hidden)
        self.reg_head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [batch, seq_len]
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        emb = self.input_proj(x)
        out = self.backbone(inputs_embeds=emb).last_hidden_state
        pred = self.reg_head(out[:, -1, :])
        return pred.squeeze(-1)
    