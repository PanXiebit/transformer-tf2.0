import time
import tensorflow as tf
from train_helper.optim import CustomSchedule
from data_process.dataset import tokenizer_pt, tokenizer_en, get_dataset
from model.transformer import Transformer
from model.multi_head_attention import create_mask

class Config():
    num_layers = 2
    d_model = 32
    dff = 32
    num_heads = 1
    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1
    max_length = 40
    buffer_size = 20000
    batch_size = 32

config = Config()

# set custom learning rate
learning_rate = CustomSchedule(config.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# loss and metrics
# from_logits 表示 y_pred 来自 logits,
# reduction=‘none’表示计算的loss是[batch]， 默认的 scalar，也就是平均值
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none")

def loss_function(real, pred):
    # 这里 real 是 target sentence 的 index，这里实际计算的就是 padding mask，也就是不考虑 padding 词的loss
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# 评价指标是 loss 和 accuracy
train_loss = tf.keras.metrics.Mean(name="train_loss") # 计算均值
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy") # 计算准确率

# training and checkpointing
transformer = Transformer(config.num_layers, config.d_model, config.num_heads, config.dff,
                          config.input_vocab_size, config.target_vocab_size, config.dropout_rate)

# training and checkpointing
# Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every n epochs.
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

# print("trainable variables:\n", transformer.trainable_variables)


@tf.function
def train_step(inp, tgt):
    tgt_inp = tgt[:, :-1]
    tgt_real = tgt[:, 1:]
    print("tgt_inp.shape:", tgt_inp.shape)
    print("tgt_real.shape:", tgt_real.shape)
    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, tgt_inp)
    print("mask.shape:", enc_padding_mask.shape, combined_mask.shape, dec_padding_mask.shape)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tgt_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        # print(tape.watched_variables())
        loss = loss_function(tgt_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # loss and metric
    train_loss(loss)
    train_accuracy(tgt_real, predictions)

train_dataset, test_dataset = get_dataset(config.max_length, config.batch_size, config.buffer_size)
EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()   # Resets all of the metric state variables.
    train_accuracy.reset_states()

    for (batch, (inp, tgt)) in enumerate(train_dataset):
        train_step(inp, tgt)

        if batch % 500 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))