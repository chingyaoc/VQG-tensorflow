import os
import tensorflow as tf
import numpy as np
import tensorflow.python.platform
from keras.preprocessing import sequence
from data_loader import *
import vgg19

class Question_Generator():
    def __init__(self, sess, conf, dataset, img_feature, train_data):
	self.sess = sess
	self.dataset = dataset
	self.img_feature = img_feature
	self.train_data = train_data
        self.dim_image = conf.dim_image
        self.dim_embed = conf.dim_embed
        self.dim_hidden = conf.dim_hidden
        self.batch_size = conf.batch_size
	self.maxlen = conf.maxlen
        self.n_lstm_steps = conf.maxlen+2
        self.model_path = conf.model_path
	if conf.is_train:
	    self.n_epochs = conf.n_epochs
	    self.learning_rate = conf.learning_rate

	self.num_train = train_data['question'].shape[0] # total number of data
	self.n_words = len(dataset['ix_to_word'].keys()) # vocabulary_size

        # word embedding
        self.Wemb = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='Wemb')
        self.bemb = tf.Variable(tf.random_uniform([self.dim_embed], -0.1, 0.1), name='bemb')

        # LSTM
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)

        # fc7 encoder
        self.encode_img_W = tf.Variable(tf.random_uniform([self.dim_image, self.dim_hidden], -0.1, 0.1), name='encode_img_W')
        self.encode_img_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.1, 0.1), name='encode_img_b')

        # feat -> word
        self.embed_word_W = tf.Variable(tf.random_uniform([self.dim_hidden, self.n_words], -0.1, 0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.random_uniform([self.n_words], -0.1, 0.1), name='embed_word_b')

    def build_model(self):
        self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        self.question = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        image_emb = tf.nn.xw_plus_b(self.image, self.encode_img_W, self.encode_img_b)        # (batch_size, dim_hidden)

        state = tf.zeros([self.batch_size, self.lstm.state_size])
        loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): 
                if i == 0:
                    current_emb = image_emb
                else:
                    tf.get_variable_scope().reuse_variables()
                    current_emb = tf.nn.embedding_lookup(self.Wemb, self.question[:,i-1]) + self.bemb

                # LSTM
                output, state = self.lstm(current_emb, state) 

                if i > 0:
                    # ground truth
                    labels = tf.expand_dims(self.question[:, i], 1) 
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat(1, [indices, labels])
                    onehot_labels = tf.sparse_to_dense(
                            concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0) 

                    # predict word
                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b) 

                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                    cross_entropy = cross_entropy * self.mask[:,i]                             

                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            self.loss = loss / tf.reduce_sum(self.mask[:,1:])

    def build_generator(self):
        self.image = tf.placeholder(tf.float32, [1, self.dim_image]) # only one image
        image_emb = tf.nn.xw_plus_b(self.image, self.encode_img_W, self.encode_img_b)

        state = tf.zeros([1, self.lstm.state_size])
        self.generated_words = []

        with tf.variable_scope("RNN"):
            output, state = self.lstm(image_emb, state)
            last_word = tf.nn.embedding_lookup(self.Wemb, [0]) + self.bemb

            for i in range(self.maxlen):
                tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(last_word, state)

                logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                max_prob_word = tf.argmax(logit_words, 1)

                last_word = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                last_word += self.bemb

                self.generated_words.append(max_prob_word)

    def train(self):
        index = np.arange(self.num_train)
        np.random.shuffle(index)

        questions = self.train_data['question'][index,:]
        img_list = self.train_data['img_list'][index]
        feats = self.img_feature[img_list,:]

        self.saver = tf.train.Saver(max_to_keep=50)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        for epoch in range(self.n_epochs):
            counter = 0

            for start, end in zip( \
                    range(0, len(feats), self.batch_size),
                    range(self.batch_size, len(feats), self.batch_size)
                    ):

                current_feats = feats[start:end]
                current_questions = questions[start:end]

                current_question_matrix = sequence.pad_sequences(current_questions, padding='post', maxlen=self.maxlen+1)
                current_question_matrix = np.hstack( [np.full( (len(current_question_matrix),1), 0), current_question_matrix] ).astype(int)

                current_mask_matrix = np.zeros((current_question_matrix.shape[0], current_question_matrix.shape[1]))
                nonzeros = np.array( map(lambda x: (x != 0).sum()+2, current_question_matrix ))
                #  +2 -> #START# and '.'

                for ind, row in enumerate(current_mask_matrix):
                    row[:nonzeros[ind]] = 1

                _, loss_value = self.sess.run([train_op, self.loss], feed_dict={
                    self.image: current_feats,
                    self.question : current_question_matrix,
                    self.mask : current_mask_matrix
                    })

                if np.mod(counter, 100) == 0:
                    print "Epoch: ", epoch, " batch: ", counter ," Current Cost: ", loss_value
                counter = counter + 1

	    if np.mod(epoch, 25) == 0:
                print "Epoch ", epoch, " is done. Saving the model ... "
		self.save_model(epoch)

    def test(self, test_image_path, model_path, maxlen):
	ixtoword = self.dataset['ix_to_word'] 

        images = tf.placeholder("float32", [1, 224, 224, 3])

        image_val = read_image(test_image_path)

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        fc7 = self.sess.run(vgg.relu7, feed_dict={images:image_val})

        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        generated_word_index = self.sess.run(self.generated_words, feed_dict={self.image:fc7})
        generated_word_index = np.hstack(generated_word_index)

        generated_sentence = ''
        for x in generated_word_index:
             if x==0:
               break
             word = ixtoword[str(x)]
             generated_sentence = generated_sentence + ' ' + word

        print ' '
        print '--------------------------------------------------------------------------------------------------------'
        print generated_sentence

    def save_model(self, epoch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.saver.save(self.sess, os.path.join(self.model_path, 'model'), global_step=epoch)
