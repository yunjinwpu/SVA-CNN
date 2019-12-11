import tensorflow as tf


class co_sp_ve_Attention(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            attention_dim=100,
            use_context_attention=None,
            use_channel_attention=True,
            use_spatial_attention=True,
            use_sp_or_ch_attention=True
    ):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        #     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        if use_context_attention:
            self.attention_hidden_dim = attention_dim
            self.attention_W = tf.Variable(
                tf.random_uniform([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                name="attention_W")
            self.attention_U = tf.Variable(
                tf.random_uniform([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                name="attention_U")
            self.attention_V = tf.Variable(
                tf.random_uniform([self.attention_hidden_dim, 1], 0.0, 1.0),
                                           name="attention_V")

            # attention layer before convolution
            self.output_att = list()
            with tf.name_scope("context_attention"):
                input_att = tf.split(self.embedded_chars, self.sequence_length, axis=1)
                for index, x_i in enumerate(input_att):
                    x_i = tf.reshape(x_i, [-1, self.embedding_size])
                    c_i = self.context_attention(x_i, input_att, index)
                    inp = tf.concat([x_i, c_i], axis=1)
                    self.output_att.append(inp)

                input_conv = tf.reshape(tf.concat(self.output_att, axis=1),
                                        [-1, self.sequence_length, self.embedding_size * 2],
                                        name="input_convolution")
            self.input_conv_expanded = tf.expand_dims(input_conv, -1)
        else:
            self.input_conv_expanded = tf.expand_dims(self.embedded_chars, -1)

        dim_input_conv = self.input_conv_expanded.shape[-2].value


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            print(i,filter_size)
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, dim_input_conv, 1, num_filters]

                #print(filter_shape)
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_conv_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                print(h.shape)

                if use_sp_or_ch_attention:
                    with tf.variable_scope('SpatialAttention', reuse=tf.AUTO_REUSE)as scope:
                        weight_decay = 0.0004
                        H = (h.shape[1])
                        W = (h.shape[2])
                        C = (h.shape[3])
                        print(H,W,C)#54 128 1
                        w_s = tf.get_variable("Spatial_Attention_w_s", [C, 1],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.orthogonal,
                                              regularizer=None)
                        b_s = tf.get_variable("Spatial_Attention_b_s", [1],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.zeros)
                        spatial_attention_fm = tf.matmul(tf.reshape(h, [-1, C]), w_s) + b_s
                        #print(spatial_attention_fm.shape)
                        spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, W * H]))
                        #print(spatial_attention_fm.shape)
                        attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
                        #print(attention.shape)
                        h = attention * h
                        #print(h.shape)

                        print("use_channel_attention")
                        with tf.variable_scope('ChannelWiseAttention', reuse=tf.AUTO_REUSE) as scope:
                            # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
                            # _, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
                            weight_decay = 0.0004
                            H = (h.shape[1])
                            W = (h.shape[2])
                            C = (h.shape[3])
                            #print(H, W, C)
                            w_s = tf.get_variable("ChannelWiseAttention_w_s", [C, C],
                                                  dtype=tf.float32,
                                                  initializer=tf.initializers.orthogonal,
                                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
                            b_s = tf.get_variable("ChannelWiseAttention_b_s", [C],
                                                  dtype=tf.float32,
                                                  initializer=tf.initializers.zeros)
                            transpose_feature_map = tf.transpose(tf.reduce_mean(h, [1, 2], keep_dims=True),
                                                                 perm=[0, 3, 1, 2])
                            channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map,
                                                                             [-1, C]), w_s) + b_s
                            channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)
                            attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (H * W),
                                                             axis=1), [-1, H, W, C])
                            #print(attention.shape)
                            h = attention * h
                            #print(h.shape)

                else:
                    h=h

                if use_spatial_attention:
                    print("spatial_attention")
                    with tf.variable_scope('SpatialAttention', reuse=tf.AUTO_REUSE)as scope:
                        weight_decay = 0.0004
                        H = (h.shape[1])
                        W = (h.shape[2])
                        C = (h.shape[3])
                        #print(H,W,C)
                        w_s = tf.get_variable("Spatial_Attention_w_s", [C, 1],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.orthogonal,
                                              regularizer=None)
                        b_s = tf.get_variable("Spatial_Attention_b_s", [1],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.zeros)
                        spatial_attention_fm = tf.matmul(tf.reshape(h, [-1, C]), w_s) + b_s
                        #print(spatial_attention_fm.shape)
                        spatial_attention_fm = tf.nn.relu(tf.reshape(spatial_attention_fm, [-1, W * H]))
                        #print(spatial_attention_fm.shape)
                        attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
                        #print(attention.shape)
                        h = attention * h
                        #print(h.shape)
                else:
                    h=h

                if use_channel_attention:
                    print("use_channel_attention")
                    with tf.variable_scope( 'ChannelWiseAttention', reuse=tf.AUTO_REUSE) as scope:
                        weight_decay = 0.0004
                        H = (h.shape[1])
                        W = (h.shape[2])
                        C = (h.shape[3])
                        print(H, W, C)
                        w_s = tf.get_variable("ChannelWiseAttention_w_s", [C, C],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.orthogonal,
                                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
                        b_s = tf.get_variable("ChannelWiseAttention_b_s", [C],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.zeros)
                        transpose_feature_map = tf.transpose(tf.reduce_mean(h, [1, 2], keep_dims=True),
                                                             perm=[0, 3, 1, 2])
                        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map,
                                                                         [-1, C]), w_s) + b_s
                        channel_wise_attention_fm = tf.nn.relu(channel_wise_attention_fm)
                        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (H * W),
                                                         axis=1), [-1, H, W, C])
                        print(attention.shape)
                        h = attention * h
                        print(h.shape)

                else:
                    h=h


                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print(pooled.shape)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print(self.scores.shape,self.input_y.shape)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.y_true=tf.argmax(self.input_y, 1)

    def context_attention(self, x_i, x, index):
        e_i = []
        c_i = []
        for output in x:
            output = tf.reshape(output, [-1, self.embedding_size])
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.sequence_length, 1)


        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.sequence_length-1, self.embedding_size])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i
