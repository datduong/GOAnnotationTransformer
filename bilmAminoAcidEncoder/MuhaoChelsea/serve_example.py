import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
import numpy as np

os.chdir('/local/datdb/MuhaoChelseaBiLmEncoder/model')
vocab_file='vocab.txt'
options_file='behm_32skip_2l.ckpt/options.json'
weight_file='behm_32skip_2l.hdf5'
token_embedding_file='vocab_embedding_32skip_2l.hdf5'

## COMMENT need to load own sequences
sequences = [['A', 'K','J','T','C','N'], ['C','A','D','A','A']]

## Serving contextualized embeddings of amino acids ================================

## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)

elmo_context_top = weight_layers('output_top_only', context_embeddings_op, l2_coef=0.0, use_top_only=True)

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())
    # Create batches of data.
    context_ids = batcher.batch_sentences(sequences)
    # Input token representations.
    elmo_context_top_ = sess.run(
        [elmo_context_top['weighted_op']],
        feed_dict={context_token_ids: context_ids}
    )
    # Output token representations.
    elmo_context_output_ = sess.run(
        [elmo_context_output['weighted_op']],
        feed_dict={context_token_ids: context_ids}
    )

print(elmo_context_output_) #contextualized embedding vector sequences (all layers)

