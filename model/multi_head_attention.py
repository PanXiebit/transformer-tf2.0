import tensorflow as tf
import numpy as np


# padding mask
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # [batch_size, 1, 1, seq_len]

# look-ahead mask
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask   # (seq_len, seq_len)

def create_mask(inp, tgt):
    """ 这里总共要计算三种 masking，
    第一种是 encoding 阶段对于 source 端的 padding  mask
    第二种是 decoder 阶段 self-attention 部分的 masked-multi-head-attention,这里不仅包括 padding mask，还包括 look ahead mask
    第三种是 decoder 阶段 encoder_ouput 和 attented target 的交互 attention,这里仅仅包括 padding mask，而且与第一阶段一致，因为是 encoder output
        作为 key 和 value 值. attented target 是 query.

    :param inp:  [batch, inp_seq_len]
    :param tgt:  [batch, tgt_seq_len]
    :return:
    """
    # 第一种masking: encoding padding mask
    enc_padding_amsk = create_padding_mask(inp)  # [batch, 1, 1, inp_seq_len]

    # 第三种masking: decoding padding mask
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)  # [batch, 1, 1, inp_seq_len]

    # 第二种masking: combined mask
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tgt)[1])   # [tgt_seq_len, tgt_seq_len] 上三角都为1，下三角（包括对角线）都为0
    dec_targrt_padding_mask = create_padding_mask(tgt)           # [batch, 1, 1, tgt_seq_len]
    combined_mask = tf.maximum(dec_targrt_padding_mask, look_ahead_mask)    # [batch, 1, tgt_seq_len, tgt_seq_len]
    # 对 [batch, 1,1, tgt_Seq_len] 第三个维度广播之后,然后再进行计算。不仅mask了future信息，还mask对应序列中的padding词
    return enc_padding_amsk, combined_mask, dec_padding_mask

def scaled_dot_product_attention(q, k, v, mask):
    """ caculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    :param q: query shape == [..., q_len, d_model]
    :param k: key shape == [..., kv_len, d_model]
    :param v: value shape == [..., kv_len, d_model]
    :param mask: Float tensor with shape broadcastable to [..., q_len, kv_len]
    :return:
        output, attention_weights. [..., q_len, kv_len]
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # [..., q_len, kv_len]
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)
    print(scaled_attention_logits.shape)
    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * 1e-9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., q_len, kv_len)
    output = tf.matmul(attention_weights, v)  # [.., q_len, d_model] ? [.., k_len, d_model]
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        :param x: [batch_size, seq_len, d_model]
        :param batch_size:
        :return:
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [batch, num_heads, -1, depth]

    def call(self, q, k, v, mask):
        """

        :param v:  [batch, q_len, d_model]
        :param k:  [batch, kv_len, d_model]
        :param q:  [batch, kv_len, d_model]
        :param mask: padding mask or look ahead mask. [..., q_len, kv_len]
        :return:
        """
        batch_size = tf.shape(q)[0]

        # Linear layers and split into heads
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        split_q = self.split_heads(q, batch_size)
        split_k = self.split_heads(k, batch_size)
        split_v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        # scaled_attention.shape == [batch, num_heads, q_len, depth]
        # attention_weights.shape == [batch, num_heads, q_len, kv_len]
        scaled_attention, attention_weights = scaled_dot_product_attention(
            split_q, split_k, split_v, mask)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # [batch, q_len, num_heads, depth]

        # concatenation
        concat_attention = tf.reshape(scaled_attention,
                                      shape=(batch_size, -1, self.d_model))  # [batch, q_len, d_model]

        #  Final linear layer
        output = self.dense(concat_attention)  # [batch, q_len, d_model]
        return output, attention_weights


# point-wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

if __name__ == "__main__":
    # def print_out(q, k, v):
    #     temp_out, temp_attn = scaled_dot_product_attention(
    #         q, k, v, None)
    #     print('Attention weights are:')
    #     print(temp_attn)  # [..., len_q, len_kv]
    #     print('Output is:')
    #     print(temp_out)  # [..., len_q, d_model]
    #
    # np.set_printoptions(suppress=True)
    # temp_k = tf.constant([[10, 0, 0],
    #                       [0, 10, 0],
    #                       [0, 0, 10],
    #                       [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    # temp_v = tf.constant([[1, 0],
    #                       [10, 0],
    #                       [100, 5],
    #                       [1000, 6]], dtype=tf.float32)  # (4, 2)
    # # This `query` aligns with the second `key`,
    # # so the second `value` is returned.
    # temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    # print_out(temp_q, temp_k, temp_v)
    #
    # # This query aligns with a repeated key (third and fourth),
    # # so all associated values get averaged.
    # temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    # print_out(temp_q, temp_k, temp_v)

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    q = tf.random.uniform((1, 62, 512))
    k = v = tf.random.uniform((1, 60, 512))
    out, attn = temp_mha(q, k, v, mask=None)
    print(out.shape, attn.shape)