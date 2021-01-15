# %%
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import sys

if not '../' in sys.path: sys.path.append('../')

import pandas as pd

from utils import data_utils
from model_config2 import config
from ved_varAttn import VarSeq2SeqVarAttnModel

import numpy as np
# %%
import matplotlib.pyplot as plt



if config['experiment'] == 'qgen':
    print('[INFO] Preparing data for experiment: {}'.format(config['experiment']))
    train_data = pd.read_csv(config['data_dir'] + 'df_qgen_train.csv')
    val_data = pd.read_csv(config['data_dir'] + 'df_qgen_val.csv')
    test_data = pd.read_csv(config['data_dir'] + 'df_qgen_test.csv')
    input_sentences = pd.concat([train_data['question'], val_data['question'], test_data['question']])
    output_sentences = pd.concat([train_data['answer'], val_data['answer'], test_data['answer']])
    true_test = test_data['answer']
    input_test = test_data['question']
    filters = '!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n'
    w2v_path = config['w2v_dir'] + 'w2vmodel_qgen.pkl'

elif config['experiment'] == 'dialogue':
    train_data = pd.read_csv(config['data_dir'] + 'df_dialogue_train.csv')
    val_data = pd.read_csv(config['data_dir'] + 'df_dialogue_val.csv')
    test_data = pd.read_csv(config['data_dir'] + 'df_dialogue_test.csv')
    input_sentences = pd.concat([train_data['line'], val_data['line'], test_data['line']])
    output_sentences = pd.concat([train_data['reply'], val_data['reply'], test_data['reply']])
    true_test = test_data['reply']
    input_test = test_data['line']
    filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
    w2v_path = config['w2v_dir'] + 'w2vmodel_dialogue.pkl'

elif config['experiment'] == 'traj':
    train_data = pd.read_csv(config['data_dir'] + 'df_traj_train.csv')
    val_data = pd.read_csv(config['data_dir'] + 'df_traj_val.csv')
    test_data = pd.read_csv(config['data_dir'] + 'df_traj_test.csv')
    input_sentences = pd.concat([train_data['question'], val_data['question'], test_data['question']])
    output_sentences = pd.concat([train_data['answer'], val_data['answer'], test_data['answer']])
    true_test = test_data['answer']
    input_test = test_data['question']
    filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
    w2v_path = config['w2v_dir'] + 'w2vmodel_traj.pkl'

elif config['experiment'] == 'disabled':
    train_data = pd.read_csv(config['data_dir'] + 'df_disabled_train.csv')
    val_data = pd.read_csv(config['data_dir'] + 'df_disabled_val.csv')
    test_data = pd.read_csv(config['data_dir'] + 'df_disabled_test.csv')
    input_sentences = pd.concat([train_data['question'], val_data['question'], test_data['question']])
    output_sentences = pd.concat([train_data['answer'], val_data['answer'], test_data['answer']])
    true_test = test_data['answer']
    input_test = test_data['question']
    filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
    w2v_path = config['w2v_dir'] + 'w2vmodel_disabled.pkl'

else:
    print('Invalid experiment name specified!')

# %%

print('[INFO] Tokenizing input and output sequences')
x, input_word_index = data_utils.tokenize_sequence(input_sentences,
                                                   filters,
                                                   config['encoder_num_tokens'],
                                                   config['encoder_vocab'])

y, output_word_index = data_utils.tokenize_sequence(output_sentences,
                                                    filters,
                                                    config['decoder_num_tokens'],
                                                    config['decoder_vocab'])

print('[INFO] Split data into train-validation-test sets')
x_train, y_train, x_val, y_val, x_test, y_test = data_utils.create_data_split(x,
                                                                              y,
                                                                              config['experiment'])

encoder_embeddings_matrix = data_utils.create_embedding_matrix(input_word_index,
                                                               config['embedding_size'],
                                                               w2v_path)

decoder_embeddings_matrix = data_utils.create_embedding_matrix(output_word_index,
                                                               config['embedding_size'],
                                                               w2v_path)
print(config['encoder_vocab'])
# Re-calculate the vocab size based on the word_idx dictionary
config['encoder_vocab'] = len(input_word_index)
config['decoder_vocab'] = len(output_word_index)
print(len(input_word_index))
# %%

model = VarSeq2SeqVarAttnModel(config,
                               encoder_embeddings_matrix,
                               decoder_embeddings_matrix,
                               input_word_index,
                               output_word_index)

# %%
enembedding = model.encoder_embeddings_matrix
deembedding = model.decoder_embeddings_matrix

print(enembedding)
print(enembedding.shape)
#
# print(output_word_index)
# print(input_word_index)
# print(input_word_index.values())
print(x_train.shape)


np.savetxt( 'encoder-embedding.txt',enembedding, fmt='%.18e', delimiter=',')
np.savetxt( 'decoder-embedding.txt',deembedding, fmt='%.18e', delimiter=',')
np.savetxt( 'y_test.txt',x_train, fmt='%d', delimiter=',')

# %%

if config['load_checkpoint'] != 0:
    checkpoint = config['model_checkpoint_dir'] + str(config['load_checkpoint']) + '.ckpt'
else:
    checkpoint = tf.train.get_checkpoint_state(os.path.dirname('models/checkpoint')).model_checkpoint_path


##
preds = model.predict(checkpoint,
                      x_test,
                      y_test,
                      true_test,
                      )



count = 100
model.show_output_sentences(preds[:count],
                            y_test[:count],
                            input_test[:count],
                            true_test[:count],
                            )

# %%

model.get_diversity_metrics(checkpoint, x_test, y_test)

# %%


