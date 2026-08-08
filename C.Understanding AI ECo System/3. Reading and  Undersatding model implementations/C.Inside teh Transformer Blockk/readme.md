``` text
GemmaDecoderLayer
    │
    ├── RMSNorm
    ├── Attention
    │     ├── Q projection
    │     ├── K projection
    │     ├── V projection
    │     ├── RoPE
    │     ├── attention computation
    │     └── output projection
    │
    ├── residual connection
    │
    ├── RMSNorm
    │
    ├── MLP
    │     ├── gate projection
    │     ├── up projection
    │     ├── activation
    │     └── down projection
    │
    └── residual connection
```
