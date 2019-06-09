from data_process.dataset import tokenizer_pt, tokenizer_en

class Config():
    num_layers = 6
    d_model = 512
    dff = 1024
    num_heads = 8
    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1
    max_length = 40
    buffer_size = 20000
    batch_size = 64