from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Bidirectional, Conv1D, \
	LSTM
from keras.layers.merge import Concatenate
from keras.layers.core import *
from keras.models import *


# ---------------------- Parameters section -------------------
# Model Hyperparameters
from WeedProject.BaseLines import attention_3d_block

embedding_dim = 50
filter_sizes = range(1,5)
num_filters = 20
dropout_prob = (0.5, 0.8)
hidden_dims = 100

# Training parameters
batch_size = 100
num_epochs = 30

# Prepossessing parameters
sequence_length = 800
max_words = 10000

# Word2Vec parameters (see train_word2vec)
min_word_count = 10
context = 10

num_classes = 3
lstm_output_size = 50
file_name = "./RunningTest.txt"
f_out = open(file_name, 'a')


INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

#
# ---------------------- Parameters end -----------------------

######Model: CLSTM with Attention#############
def build_clstm_attenion(model_input, z):
	conv_blocks = []
	for i, sz in enumerate(filter_sizes):
		conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
		# conv = MaxPooling1D(pool_size=2)(conv)
		# conv = Flatten()(conv)
		conv_blocks.append(conv)
	z = Concatenate(axis=1)(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

	# attention before LSTM
	# z = attention_3d_block(z)
	lstm_out = LSTM(lstm_output_size, return_sequences=True)(z)
	# attention after LSTM
	attention_mul = attention_3d_block(lstm_out)
	attention_mul = Flatten()(attention_mul)
	model_output = Dense(3, activation='softmax')(attention_mul)
	model = Model(model_input, model_output)
	return model