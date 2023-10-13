# calculates for 13B model. Change the constraints for 7B model accordingly!
def flops_calc(batch_size, seq_len):

    N = batch_size
    T = seq_len
    d_model = 5120  # Model dimension
    d_ff = 13824  # Feed-forward layer dimension
    n_heads = 1  # Number of attention heads
    n_layers = 40  # Number of decoder layers
    vocab_size = 32016  # Vocabulary size

    # Embedding FLOPs
    embedding_flops = 0

    # Self Attention FLOPs for one layer
    attention_flops = 4 * 2 * N * T * d_model * d_model

    # MLP FLOPs for one layer
    mlp_flops = 2 * N * T * (d_model * d_ff + d_ff * d_model)

    # LayerNorm FLOPs for one layer (mean, variance, and normalization)
    layernorm_flops = N * T * d_model * 2

    # Total FLOPs for one Decoder Layer
    one_layer_flops = attention_flops + mlp_flops + layernorm_flops

    # Total FLOPs for 32 Decoder Layers
    total_decoder_flops = n_layers * one_layer_flops

    # lm_head Linear layer FLOPs
    lm_head_flops = 2 * N * T * d_model * vocab_size

    # Total FLOPs
    total_flops = embedding_flops + total_decoder_flops + lm_head_flops


    return total_flops
