'''
 This file contains the class of a variational auto-encoder architecture, for processing paraphrased sentences

'''
import tensorflow as tf
import numpy as np

import tools

STOP = '<STOP>'
SIMILARITY_PORTION_RATIO = 0.5

class Variational_Autoencoder:
    """
    This is the class for building a variation auto-encoder. It should takes in sentences (represented by word IDs) and
    reconstruct the input sentences.
    This class support two modes: training mode, and linear interpolation mode.
    """
    def __init__(self, nlp, id2voc_dict, word_embedding_size, vocab_size, RNN_HIDDEN_SIZE, latent_code_batch_size, LATENT_DIM, max_message_length, LEARNING_RATE, ENC_DROPOUT_RATE,
                 SAVE_DIR, LOAD_FROM_SAVE, mode, Train = False, USE_SPACY=False):
        """

        :param word_embedding_size:
        :param vocab_size:
        :param RNN_HIDDEN_SIZE:
        :param max_message_length:
        """
        self.id2voc_dict= id2voc_dict
        self.vocab_size = vocab_size
        self.latent_code_batch_size = latent_code_batch_size
        self.word_embedding_size = word_embedding_size
        self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE
        self.max_message_length = max_message_length
        self.LEARNING_RATE = LEARNING_RATE
        self.enc_dropout_rate = ENC_DROPOUT_RATE
        self.save_dir = SAVE_DIR
        self.load_from_save = LOAD_FROM_SAVE
        self.mode = mode
        self.Train = Train
        self.SAVE_DIR = SAVE_DIR
        self.LATENT_DIM = LATENT_DIM


        #with tf.name_scope('Data_input'):
        self.seq_id = tf.placeholder(tf.int32, [None, self.max_message_length], name='input_message_ID')
            #this None corresponds to batch_size

        self.kl_rate = tf.placeholder_with_default(1.0, (), name='kl_rate')

        # create embedding
        with tf.variable_scope('Word_emb'):
            if USE_SPACY == False:
                self.embed_matrix = tf.get_variable(name='embedding_matrix',
                                                   shape=[self.vocab_size, self.word_embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
                #seq_emd = tf.nn.embedding_lookup(embed_matrix, self.seq_id, name='input_message_embedding')
            else:
                # self.embed_matrix = tf.zeros(name='embedding_matrix',
                #                                    shape=[self.vocab_size, self.word_embedding_size])
                # for i in range(self.vocab_size):
                #     self.embed_matrix[i,:] = nlp(id2voc_dict[i]).vector
                self.embed_matrix = tf.Variable(tf.constant(0.0, shape=[vocab_size, self.word_embedding_size]),
                                trainable=False, name="embedding_matrix")
                embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, self.word_embedding_size])
                embedding_init = self.embed_matrix.assign(embedding_placeholder)

        if mode == "both":
            # create the encoder
            self.enc_latent_code, self.mean, self.log_var = self.build_encoder(self.seq_id, self.enc_dropout_rate)
            # create the decoder
            self.decoder_input = self.enc_latent_code
            self.predictions, self.probabilities, self.scores_3d = self.build_decoder(self.decoder_input)
        elif mode == "encode":
            self.enc_latent_code, self.mean, self.log_var = self.build_encoder(self.seq_id, self.enc_dropout_rate)
        elif mode == "decode":
            self.latent_code = tf.placeholder(dtype=tf.float32, shape=[None, self.LATENT_DIM], name='latent_code')
            self.decoder_input = self.latent_code
            self.predictions, self.probabilities, self.scores_3d = self.build_decoder(self.decoder_input)
        else:
            print ("Error in creating the autoencoder graph!")

        if self.Train:
            # create the training op with loss (from the decoder output)
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.train_op, self.data_loss, self.KL_loss, self.similarity_mse, self.difference_loss =\
                                                                            self.build_trainer(self.scores_3d, self.seq_id, self.enc_latent_code,
                                                                            self.mean, self.log_var, self.LEARNING_RATE)

        # might need saver
        with tf.name_scope("SAVER"):
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

        # start session and initialize all variables
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        if USE_SPACY:
            embedding_arr = np.zeros([self.vocab_size, self.word_embedding_size])
            for i in range(self.vocab_size):
                embedding_arr[i,:] = nlp(id2voc_dict[i]).vector
            self.sess.run(embedding_init, feed_dict={embedding_placeholder: embedding_arr})
        if self.load_from_save:
            print("Loading from save: ")
            #self.load_scope_from_save(self.SAVE_DIR, self.sess, 'Word_emb') # load word embedding
            if mode == "encode":
                self.load_scope_from_save(self.SAVE_DIR, self.sess, 'Encoder')
            elif mode == "decode":
                self.load_scope_from_save(self.SAVE_DIR, self.sess, 'Decoder')
            else:
                print ("Mode is neither encoder nor decode!")





    def build_encoder(self, seq_id, dropout_rate):
        """
            Build the encoder.
            Args:
                seq_id: a batch of sequences, consisting of word IDs
            Return:
                encoder_output = encoded_out_put,
                if with out variational component, it is input to the decoder of size (batch_size, 1, hidden_size)

        """
        with tf.variable_scope('Encoder'):
            seq_emd = tf.nn.embedding_lookup(self.embed_matrix, seq_id, name='input_message_embedding') # (batchsize * 300)
            # could implement drop out here for the embedding
            enc_lstm = tf.contrib.rnn.LSTMCell(self.RNN_HIDDEN_SIZE)
            encoded_lstm_output, encoded_state = tf.nn.dynamic_rnn(cell=enc_lstm,
                                                              inputs=seq_emd,
                                                              dtype=tf.float32)
            if self.Train: # dropout only added for training stage, not testing
                encoded_output = tf.nn.dropout(encoded_lstm_output[:, -1, :], (1.0 - dropout_rate))
            else:
                encoded_output = encoded_lstm_output[:, -1, :]
            # encoded_output is not necessarily the encoder_output, which is the latent code, sampled using reparametrization
            # reparametrization:
            mu = self.fully_connected_layer(encoded_output, self.RNN_HIDDEN_SIZE, self.LATENT_DIM,
                                               None, True, name='mean')
            log_var = self.fully_connected_layer(encoded_output, self.RNN_HIDDEN_SIZE, self.LATENT_DIM,
                                                      None, True, name = 'log_variance')
            epsilon = tf.random_normal(tf.shape(mu), stddev=1, mean=0)
            latent_code = mu + tf.multiply(epsilon, tf.sqrt(tf.exp(log_var)))
        return latent_code, mu, log_var



    def build_decoder(self, decoder_input):
        """

        :param decoder_input:
        without variational component, it is the encoder_output, the output at the last time step
        of the encoding network, of size (batch, 1, hidden_size)

        :return:
        predictions: batch * max_length, each entry is a workd ID
        probabilities: batch_size * maximum_length * vocab_size, each entry is a probability on the vocab_index
        scores_3d: batch_size * maximum_length * vocab_size, each entry is a unnormalized log likelihood

        """
        with tf.variable_scope('Decoder'):
            encoded_output_tile = tf.tile(tf.reshape(decoder_input, [-1, 1, self.LATENT_DIM]),
                                      [1, self.max_message_length, 1])
            dec_lstm = tf.contrib.rnn.LSTMCell(self.RNN_HIDDEN_SIZE)
            decoded_output, decoded_state = tf.nn.dynamic_rnn(cell=dec_lstm,
                                                              inputs=encoded_output_tile,
                                                              #initial_state=encoded_state,
                                                              dtype=tf.float32)
            # predictions
            with tf.name_scope('Predictions'):
                output_w = tf.get_variable(name='output_weights',
                                           shape=[self.RNN_HIDDEN_SIZE, self.word_embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
                output_b = tf.get_variable(name='output_bias',
                                           shape=[self.word_embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
                decoded_output_2d = tf.reshape(decoded_output,
                                               [-1, self.RNN_HIDDEN_SIZE])  # (batch_size*max_len, RNN_HIDDEN_SIZE)
                word_output_2d = tf.matmul(decoded_output_2d, output_w) + output_b  # (batch_size*max_len, word_embedding_size)
                scores_2d = tf.matmul(word_output_2d, self.embed_matrix, transpose_b=True) #(batch_size * max_len, vocab_size)
                scores_3d = tf.reshape(scores_2d,
                                       [-1, self.max_message_length, self.vocab_size])  # (batch_size, max_len, vocab_size)
                probabilities = tf.nn.softmax(scores_3d, dim=-1, name='probabilities_over_vocab')
                # batch_size * maximum_length * vocab_size
                predictions = tf.argmax(probabilities, axis=2)  # batch * max_length, each entry is a word ID
        return predictions, probabilities, scores_3d

    def build_trainer(self, scores_3d, seq_id, latent_code, mean, log_var, learning_rate):
        """

        :param scores_3d:
        :param seq_id:
        :param latent_code: the dimension is (batch_size * 2, latent_dim)
        :param mean:
        :param log_var:
        :param learning_rate:
        :return:
        """
        with tf.variable_scope('Loss'):
            data_loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_3d, labels=seq_id, name='Reconstruction_loss')
            data_loss = tf.reduce_mean(data_loss_batch)
            # KL divergence loss:
            KL_loss_batch = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            KL_loss = tf.reduce_mean(KL_loss_batch)
            # similarity_loss
            similarity_portion = int(self.LATENT_DIM * SIMILARITY_PORTION_RATIO)
            similatrity_mse = tf.losses.mean_squared_error(latent_code[:self.latent_code_batch_size, :similarity_portion],
                                 latent_code[self.latent_code_batch_size:, :similarity_portion])
            distance = tf.abs(latent_code[:self.latent_code_batch_size, :similarity_portion] -
                                 latent_code[self.latent_code_batch_size:, :similarity_portion])
            difference_loss = - tf.reduce_mean(1.0 - tf.nn.relu(1.0 - distance))

            # similatrity_measure = tf.reduce_mean((latent_code[:self.latent_code_batch_size, :self.LATENT_DIM/2]
            #                                        - latent_code[self.latent_code_batch_size:, :self.LATENT_DIM/2])**2)



            #loss = tf.reduce_mean(data_loss + KL_loss + similarity_loss + distinctive_loss)
            loss = tf.reduce_mean(data_loss + KL_loss + similatrity_mse + difference_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss, global_step=self.global_step)
            return optimizer, data_loss, KL_loss, similatrity_mse, difference_loss


    def fully_connected_layer(self, input_tensor, input_size, output_size, activation=None, include_bias=True,
                                  name=None):
        with tf.name_scope(name):
            fc_w = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.1))
            fc_b = tf.Variable(tf.zeros([output_size]))
            if (not activation) and include_bias:
                fc_output = tf.matmul(input_tensor, fc_w) + fc_b
        return fc_output


    #internal use
    def load_scope_from_save(self, SAVE_DIR, sess, scope):
        """
        This function load variables in one scope from a directory of saved models.
        This is copied from David Donahue 2017, baseline_model_func
        :param SAVE_DIR: the directory name where the model trained is saved
        :param sess: current session
        :param scope: name scope of variables to be loaded
        :return:
        """
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        #print (vars)
        #print (scope)
        assert len(vars) > 0
        if sess is None:
            sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        saver = tf.train.Saver(max_to_keep=10, var_list=vars)
        # Restore model from previous save.
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint found!")
            return -1

        return sess


    # external use
    def train(self, NUM_EPOCH, train_set, BATCH_SIZE) :
        """
        This is a function for external call from run.py to train
        :param NUM_EPOCH:
        :param train_set:
        :param BATCH_SIZE:
        :return: return reconstructed_train_ID, with size Num_train_set * max_length, a matrix of word IDs (integers)
        """
        for i in range(NUM_EPOCH):
            print ("Epoch: %s" % i)
            kl_rate_now = min((1.0  * i / (NUM_EPOCH * 0.75)), 1.0 ) # the last 25% of the epochs are trained with full KL
            train_batch_generator = tools.BatchGenerator(train_set, BATCH_SIZE)
            if i == NUM_EPOCH - 1:  # record training example at the final epoch
                reconstructed_train_ID = []
            total_to_predict = 0.0
            correct_predict = 0.0
            for batch_index, train_batch in enumerate(train_batch_generator.generate_batches()):
                # print ('The dimension of train batch is ')
                # print np.shape(train_batch)
                #train_batch = np.reshape(train_batch, (BATCH_SIZE*2, self.max_message_length))
                #print (train_batch.shape)
                train_batch_1 = train_batch[:, :self.max_message_length]
                train_batch_2 = train_batch[:, self.max_message_length:]
                assert train_batch_1.shape[1] == train_batch_2.shape[1]
                train_batch = np.concatenate([train_batch_1, train_batch_2], 0)

                ##### DEBUG ############
                # if i == NUM_EPOCH - 1:
                #     print ("Batch index %s" % batch_index)
                #
                #     str_arr = tools.convert_id2str(train_batch_1, self.id2voc_dict)
                #     print ("train_batch_1_shape:")
                #     print (train_batch_1.shape)
                #     for x in range(train_batch_1.shape[0]):
                #         print (tools.print_str2sentence(str_arr[x], STOP))
                #
                #     str_arr = tools.convert_id2str(train_batch_1, self.id2voc_dict)
                #     print ("train_batch_2_shape: ")
                #     print (train_batch_2.shape)
                #     for x in range(train_batch_2.shape[0]):
                #         print (tools.print_str2sentence(str_arr[x], STOP))
                ##### DEBUG ############
                _, batch_data_loss, batch_KL_loss, batch_similarity_mse, batch_difference_loss, reconstructed_seq_ID_batch = self.sess.run([self.train_op,
                                                                                               self.data_loss, self.KL_loss,
                                                                                            self.similarity_mse, self.difference_loss,
                                                                                            self.predictions],
                                                                                        feed_dict={self.seq_id: train_batch,
                                                                                                   self.kl_rate: kl_rate_now})
                for j in range(train_batch.shape[0]):
                    for k in range(train_batch.shape[1]):
                        if self.id2voc_dict[train_batch[j][k]] == STOP:
                            break
                        total_to_predict += 1.0
                        if train_batch[j][k] == reconstructed_seq_ID_batch[j][k]:
                            correct_predict += 1.0
                if i == NUM_EPOCH - 1:
                    reconstructed_train_ID.append(reconstructed_seq_ID_batch)
                    ##### DEBUG ############
                    # print ("Batch index %s" % batch_index)
                    # str_arr = tools.convert_id2str(train_batch, self.id2voc_dict)
                    # print (train_batch.shape)
                    # for x in range(train_batch.shape[0]):
                    #     if x == train_batch.shape[0]/2 :
                    #         print ("---------------")
                    #     print (tools.print_str2sentence(str_arr[x], STOP))
                    ##### DEBUG ############
            print ("The batch training accuracy is %s" % (correct_predict / total_to_predict))
            print ("The batch data loss is: %s" % batch_data_loss)
            print ("The batch KL loss is: %s" % batch_KL_loss)
            print ("The batch similarity mse is %s" %batch_similarity_mse)
            print ("The batch difference loss is %s" %(-batch_difference_loss))
            self.saver.save(self.sess, self.save_dir, global_step=self.global_step)
        reconstructed_train_ID = np.concatenate(reconstructed_train_ID, axis=0)  # Num_train_set * max_length
        return reconstructed_train_ID



    # external use
    def encode(self, input_ID, batch_size = None):
        """

        This function takes in sentences (in word ID) and convert them to latent codes

        :param input_ID: integer matrix of input sentences, of size (num_sentences * max_length), word IDs
        :return: latent code, matrix of size (num_sentences * latent space dimension)
        """
        assert (self.mode == "both" or self.mode == "encode")
        if batch_size is None:
            batch_size = input_ID.shape[0]
        encoded_latent_code = []
        encode_batch_generator = tools.BatchGenerator(input_ID, batch_size)
        for batch_index, input_batch in enumerate(encode_batch_generator.generate_batches()):
            latent_code_batch = self.sess.run(self.enc_latent_code, feed_dict={self.seq_id: input_batch})
            encoded_latent_code.append(latent_code_batch)
            encoded_latent_code = np.concatenate(encoded_latent_code, axis = 0)
        return encoded_latent_code

    # external use
    def decode(self, latent_code, batch_size = None):
        """
        This function takes in latent codes and convert them to sentences (list of word IDs)
        :param latent_code: matrix of size (num_sentences * latent_code_dimension)
        :param batch_size: num_sentences to be decoded
        :return: reconstructed_ID, reconstructed sentences of size (num_senteces * max_length)
        """
        assert (self.mode == "both" or self.mode == "decode")
        if batch_size is None:
            batch_size = latent_code.shape[0]
        reconstructed_ID = []
        decode_batch_generator = tools.BatchGenerator(latent_code, batch_size)
        for batch_index, latent_batch in enumerate(decode_batch_generator.generate_batches()):
            reconstructed_batch = self.sess.run(self.predictions, feed_dict = {self.latent_code: latent_code})
            reconstructed_ID.append(reconstructed_batch)
        reconstructed_ID = np.concatenate(reconstructed_ID, axis=0)
        return reconstructed_ID


    # external use
    def reconstruct(self, input_ID, batch_size=None, validation = False):
        """

        :param input_ID: matrix of integers of size (batch_size * max_length), representing
        :param batch_size:
        :return:
        """
        if batch_size == None:
            batch_size = input_ID.shape[0]
        input_batch_generator = tools.BatchGenerator(input_ID, batch_size)
        # when this batch_size is equal to argument, the for loop only run one iteration
        reconstructed_ID = []
        for batch_index, input_batch in enumerate(input_batch_generator.generate_batches()):
            # if validation == False:
            #     print batch_index
            #     print input_batch
            batch_data_loss, batch_KL_loss, reconstructed_ID_batch = self.sess.run([self.data_loss, self.KL_loss, self.predictions],
                                                                     feed_dict={self.seq_id: input_batch})
            if batch_index % 5 == 0 and validation == True: # we are in the validation stage
                print ("Validation iteration: %s" % batch_index)
                print ("Validation data loss: %s" % batch_data_loss)
                print ("Validation KL loss: %s" % batch_KL_loss)
            reconstructed_ID.append(reconstructed_ID_batch)
        #if validation == True:
        reconstructed_ID = np.concatenate(reconstructed_ID, axis=0)
        return reconstructed_ID



    def print_here(self):
        print ("HERE")