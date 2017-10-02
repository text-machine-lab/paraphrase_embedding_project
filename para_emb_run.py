
import para_emb_model
import tensorflow as tf
import numpy as np
import pickle
import os
import time
import gensim
import spacy
import tools

DATA_DIR = './ppdb_data_dir/'
SAVE_DIR= './paraphrases_model_parameters/'
FILE_NAME = 'ppdb-1.0-s-phrasal'
DICT_DIR = './paraphrase_dictionaries/'
MAX_NUM_PAIRS = 200
MAX_NUM_MESSAGES_UNRESRICTED = False
RESTORE_DICT = False
LOAD_FROM_SAVE = False
TRAIN_FROM_SAVE = False # To be experimented
USE_SPACY = True
MAX_MESSAGE_LENGTH = 5
MESSAGE_LENGTH_UNRESTRICTED = False
DELIMITER = ' ||| '
STOP = '<STOP>'
RNN_HIDDEN_SIZE = 2000
LATENT_DIM = 1000
WORD_EMBEDDING_SIZE = 300
LEARNING_RATE = 0.01
ENC_DROPOUT_RATE = 0.1
TRAINING_FRACTION = 0.7
NUM_EPOCH = 500
BATCH_SIZE = 40
NUM_SELECTED_TO_PRINT = 15

print ("Begin Processing Data ...")

#nlp = None
# load spacy
#if USE_SPACY:
print ("Loading Spacy ...")
tic = time.clock()
nlp = spacy.load('en_core_web_sm') # there are other spacy embeddings available, en_core_web_md
toc = time.clock()
print ("Loading Spacy finished! It takes %s seconds." %(toc - tic))

# load the data file, process and tokenize the words
print("Loading data file ...")
tic = time.clock()

all_pairs, all_phrase_1_lengths, all_phrase_2_lengths, num_all_pair, max_message_length, avg_message_length \
    = tools.construct_pairs_list(DATA_DIR, FILE_NAME, DELIMITER, nlp, STOP, MAX_NUM_MESSAGES_UNRESRICTED,
                            MAX_NUM_PAIRS, MAX_MESSAGE_LENGTH, MESSAGE_LENGTH_UNRESTRICTED)

# all_pairs is list of list of list (num_pars, 2, length)
# for i in range(len(all_pairs)):
#     print (all_pairs[i][0] + ["\t\t\t"] + all_pairs[i][1])
print ("In total there are %s pairs." % num_all_pair)
print ("The longest message is %s-token-long." % max_message_length)
print ("The average message is %s-token-long." % avg_message_length)
toc = time.clock()
print ("Finished loading data file. It takes: %s seconds!" % (toc - tic))



# use the messages to build a dictionary
print ("Building the vocab dictionary ...")
if RESTORE_DICT:
    print ("Restore old dictionary ... ")
    voc2id_dict = pickle.load(open(os.path.join(DICT_DIR, 'voc2id_dict.pkl'),'rb'))
    id2voc_dict = {a: b for b, a in voc2id_dict.items()}
else:
    print ("Build new dictionary ...")
    all_messages = []
    for pair in all_pairs:
        all_messages.append(pair[0])
        all_messages.append(pair[1])
    print (all_messages[0])
    voc2id_dict = gensim.corpora.Dictionary(documents=all_messages).token2id
    id2voc_dict = {a: b for b, a in voc2id_dict.items()}
    # put null string '' at the zero index, and move the original word at index zero to last
    last = len(voc2id_dict)
    id2voc_dict[last] = id2voc_dict[0]
    voc2id_dict[id2voc_dict[0]] = last
    voc2id_dict[''] = 0
    id2voc_dict[0] = ''
    pickle.dump(voc2id_dict, open(os.path.join(DICT_DIR, 'voc2id_dict.pkl'), 'wb'))
    vocab_size = len(voc2id_dict)
print ("Finished constructing dictionary, with %s vocabularies" % vocab_size)
print ("")


# convert all_messages from a list of string sub-lists, to an integer matrix (M, L)
# M = Number of pairs * 2, L = Maximum length of message. Each entry is a vocab ID
print ("Converting messages in string to messages in word ID ...")
all_messages_ID = tools.convert_messages_str2id(all_messages, voc2id_dict, max_message_length)
all_pairs_ID = np.reshape(all_messages_ID, (num_all_pair, max_message_length * 2))

# test that the conversion from pairs to pairs_ID is correct
print ("Checking the conversion is correct ...")
for i in range(all_pairs_ID.shape[0]):
    pair_ID = all_pairs_ID[i]
    phrase_1_ID = all_pairs_ID[i][0:max_message_length-1]
    phrase_2_ID = all_pairs_ID[i][max_message_length:]
    #assert (phrase_1_ID + phrase_2_ID) == all_pairs_ID[i]
    message_1 = all_pairs[i][0]
    message_2 = all_pairs[i][1]
    for j, token_ID in enumerate(phrase_1_ID):
        if j < len(message_1):
            assert message_1[j] == id2voc_dict[token_ID]
        else:
            assert id2voc_dict[token_ID] == id2voc_dict[0]
            assert token_ID == 0

    for j, token_ID in enumerate(phrase_2_ID):
        if j < len(message_2):
           assert message_2[j] == id2voc_dict[token_ID]
        else:
            assert id2voc_dict[token_ID] == id2voc_dict[0]
            assert token_ID == 0
print ("The conversion is correct :) ")
print ("")

with tf.Graph().as_default() as autoencoder_graph:
    latent_code_batch_size = BATCH_SIZE
    auto_encoder = para_emb_model.Variational_Autoencoder(nlp, id2voc_dict, WORD_EMBEDDING_SIZE, vocab_size,
                                                          RNN_HIDDEN_SIZE, latent_code_batch_size, LATENT_DIM,
                                                     max_message_length, LEARNING_RATE, ENC_DROPOUT_RATE,
                                                     SAVE_DIR,
                                                     LOAD_FROM_SAVE = False,
                                                     mode = "both", Train = True,
                                                     USE_SPACY=USE_SPACY)

print ('Finished building the model.')
print ("")


# trainining
print ('Start training ...')
if True: # to take care of the indentation
    num_train_set = int(num_all_pair * TRAINING_FRACTION)
    train_set = all_pairs_ID[:num_train_set,:]
    reconstructed_train_ID = auto_encoder.train(NUM_EPOCH, train_set, BATCH_SIZE) # the most impt training step
    reconstructed_train_ID_selected = reconstructed_train_ID[:NUM_SELECTED_TO_PRINT]
    reconstructed_train_str_selected = tools.convert_id2str(reconstructed_train_ID_selected, id2voc_dict)

    print ('Training finished')
    print ('Take a look at some reconstructed sequences from training: ')
    for i in range(len(reconstructed_train_str_selected)):
        print (i)
        print ("Original: %s" % tools.print_str2sentence(all_messages[i], STOP))
        print ("Reconstructed: %s" %tools.print_str2sentence(reconstructed_train_str_selected[i], STOP))
        print ('')