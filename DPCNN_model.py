from keras import Input, Model
from keras.layers import BatchNormalization, regularizers, Embedding, SpatialDropout1D, Conv1D, PReLU, MaxPooling1D, \
   add, GlobalMaxPooling1D, Dense, Dropout



def CNN(x):
   block = Conv1D(filter_nr, kernel_size=filter_size, padding=same, activation=linear,
               kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
   block = BatchNormalization()(block)
   block = PReLU()(block)
   block = Conv1D(filter_nr, kernel_size=filter_size, padding=same, activation=linear,
               kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block)
   block = BatchNormalization()(block)
   block = PReLU()(block)
   return block


def DPCNN():
   filter_nr = 64 #滤波器通道个数
   filter_size = 3 #卷积核
   max_pool_size = 3 #池化层的pooling_size
   max_pool_strides = 2 #池化层的步长
   dense_nr = 256 #全连接层
   spatial_dropout = 0.2
   dense_dropout = 0.5
   train_embed = False
   conv_kern_reg = regularizers.l2(0.00001)
   conv_bias_reg = regularizers.l2(0.00001)

   comment = Input(shape=(maxlen,))
   emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
   emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

   #region embedding层
   resize_emb = Conv1D(filter_nr, kernel_size=1, padding=same, activation=linear,
               kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
   resize_emb = PReLU()(resize_emb)
   #第一层
   block1 = CNN(emb_comment)
   block1_output = add([block1, resize_emb])
   block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)
   #第二层
   block2 = CNN(block1_output)
   block2_output = add([block2, block1_output])
   block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)
   #第三层
   block3 = CNN(block2_output)
   block3_output = add([block3, block2_output])
   block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)
   #第四层
   block4 = CNN(block3_output)
   block4_output = add([block4, block3_output])
   block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)
   #第五层
   block5 = CNN(block4_output)
   block5_output = add([block5, block4_output])
   block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)
   #第六层
   block6 = CNN(block5_output)
   block6_output = add([block6, block5_output])
   block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)
   #第七层
   block7 = CNN(block6_output)
   block7_output = add([block7, block6_output])
   output = GlobalMaxPooling1D()(block7_output)
   #全连接层
   output = Dense(dense_nr, activation=linear)(output)
   output = BatchNormalization()(output)
   output = PReLU()(output)
   output = Dropout(dense_dropout)(output)
   output = Dense(6, activation=sigmoid)(output)

   model = Model(comment, output)
   model.summary()
   model.compile(loss=binary_crossentropy,
               optimizer=optimizers.Adam(),
               metrics=[accuracy])
   return model