import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates  # broadcast


def positional_encoding(position, d_model):
    angle_rads = get_angles(pos=np.arange(position)[:, np.newaxis],
                            i=np.arange(d_model)[np.newaxis, :],
                            d_model=d_model)  # [pos, d_model]

    # applay sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # applay cos to odd indices in the array; 2i +1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == "__main__":
    # a = np.ones((2,3))
    # sin = np.sin(a[:, 0::2])
    # cos = np.cos(a[:,1::2])
    # print(a)
    # print(sin)
    # print(cos)
    # pe = np.concatenate([sin, cos], 1)
    # print(pe)
    # print(pe[np.newaxis, ...])  # [1, pos, d_model]

    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()