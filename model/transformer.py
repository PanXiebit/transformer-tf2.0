import tensorflow as tf
from model.encoder_decoder import Encoder, Decoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tgt, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask) # [batch, input_seq_len, d_model]

        # dec_output.shape == (batch, tgt_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tgt, enc_output, training,
                                                     look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

if __name__ == "__main__":
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000)

    temp_input = tf.random.uniform((64, 62))  # [batch, inp_seq_len]
    temp_target = tf.random.uniform((64, 26)) # [batch, tgt_seq_len]

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)