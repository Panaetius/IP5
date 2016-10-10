import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d

def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X

shape = (50, 50)
board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])

initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

with tf.Session() as session:
    initial_board_values = session.run(initial_board)
    X = session.run(board_update, feed_dict={board: initial_board_values})[0]
    fig = plt.figure()
    plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
    import matplotlib.animation as animation
    def game_of_life(*X):
        X = session.run(board_update, feed_dict={board: X})[0]
        plot.set_array(X)
        return plot,

    ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True, fargs=X)
    plt.show()