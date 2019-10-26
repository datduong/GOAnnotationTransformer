
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import numpy as np 

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_transformers.modeling_bert import *
from pytorch_transformers.tokenization_bert import BertTokenizer

import KmerModel.TokenClassifier as TokenClassifier

class BertSelfAttentionDistance(nn.Module):
  def __init__(self, config):
    super(BertSelfAttentionDistance, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.output_attentions = config.output_attentions

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size ## compute for all the heads in 1 single function call

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    ## add in the distance-type weights. for example, distance 0-->0 , [1-5]-->1 , [6-10]-->2
    self.distance_vector = nn.Embedding(config.distance_type,config.hidden_size,padding_idx=0) ## distance 0 --> type 0 --> vector 0
  
  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask, head_mask=None, word_word_relation=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


    # now we have to add in the extra distance-type
    # @query_layer is head x batch x word x dim
    word_dot_distance = torch.matmul(query_layer,self.distance_vector.weight.transpose(0,1))
    # extract hidden_dot_distance using word-word-position distance matrix
    # use torch.gather https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    # using torch.gather in forward pass only is fast, why backward pass is very slow. why ? ... using the "(z==1).float()" helps a lot.
    # print (query_layer.shape)
    # print (word_dot_distance.shape)
    # print (word_word_relation.unsqueeze(1).shape)
    mask_out = word_word_relation == 1
    mask_out = mask_out.unsqueeze(1).float()
    word_word_distance_att = torch.gather( word_dot_distance, dim=3, index=word_word_relation.unsqueeze(1) ) * mask_out


    ## add to the traditional @attention_scores
    # before... attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # want to try without torch.gather
    # print (attention_scores.shape)
    # print (word_word_distance_att.shape)
    attention_scores = (attention_scores + word_word_distance_att) / math.sqrt(self.attention_head_size)

    # Apply the attention mask is (precomputed for all layers in BertModelDistance forward() function)
    attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    return outputs


class BertAttentionDistance(nn.Module):
  def __init__(self, config):
    super(BertAttentionDistance, self).__init__()
    self.self = BertSelfAttentionDistance(config)
    self.output = BertSelfOutput(config)
    self.pruned_heads = set()

  def prune_heads(self, heads):
    if len(heads) == 0:
      return
    mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
    heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
    for head in heads:
      # Compute how many pruned heads are before the head and move the index accordingly
      head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
      mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()

    # Prune linear layers
    self.self.query = prune_linear_layer(self.self.query, index)
    self.self.key = prune_linear_layer(self.self.key, index)
    self.self.value = prune_linear_layer(self.self.value, index)
    self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    # Update hyper params and store pruned heads
    self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  def forward(self, input_tensor, attention_mask=None, head_mask=None, word_word_relation=None):
    self_outputs = self.self(input_tensor, attention_mask, head_mask, word_word_relation)
    attention_output = self.output(self_outputs[0], input_tensor)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


class BertLayerDistance(nn.Module):
  def __init__(self, config):
    super(BertLayerDistance, self).__init__()
    self.attention = BertAttentionDistance(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask=None, head_mask=None, word_word_relation=None):
    attention_outputs = self.attention(hidden_states, attention_mask, head_mask, word_word_relation)
    attention_output = attention_outputs[0]
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
    return outputs


class BertEncoderDistance(nn.Module):
  def __init__(self, config):
    super(BertEncoderDistance, self).__init__()
    self.output_attentions = config.output_attentions
    self.output_hidden_states = config.output_hidden_states
    self.layer = nn.ModuleList([BertLayerDistance(config) for _ in range(config.num_hidden_layers)])

  def forward(self, hidden_states, attention_mask=None, head_mask=None, word_word_relation=None):
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
      if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], word_word_relation)
      hidden_states = layer_outputs[0]

      if self.output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)

    # Add last layer
    if self.output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
      outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
      outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertModelDistance(BertPreTrainedModel):
  r"""
  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
    **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
      Sequence of hidden-states at the output of the last layer of the model.
    **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
      Last layer hidden-state of the first token of the sequence (classification token)
      further processed by a Linear layer and a Tanh activation function. The Linear
      layer weights are trained from the next sentence prediction (classification)
      objective during Bert pretraining. This output is usually *not* a good summary
      of the semantic content of the input, you're often better with averaging or pooling
      the sequence of hidden-states for the whole input sequence.
    **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
      list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
      of shape ``(batch_size, sequence_length, hidden_size)``:
      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    **attentions**: (`optional`, returned when ``config.output_attentions=True``)
      list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

  Examples::

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModelDistance.from_pretrained('bert-base-uncased')
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

  """
  def __init__(self, config):
    super(BertModelDistance, self).__init__(config)

    self.embeddings = TokenClassifier.BertEmbeddingsAA(config)
    self.embeddings_label = TokenClassifier.BertEmbeddingsLabel(config) ## label takes its own emb layer
    self.encoder = BertEncoderDistance(config)
    self.pooler = BertPooler(config)

    self.init_weights()

    ## fix it back to 0 
    # self.encoder.BertSelfAttentionDistance.distance_vector.weight[0]=0


  def _resize_token_embeddings(self, new_num_tokens):
    old_embeddings = self.embeddings.word_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    self.embeddings.word_embeddings = new_embeddings
    return self.embeddings.word_embeddings

  def _prune_heads(self, heads_to_prune):
    """ Prunes heads of the model.
      heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
      See base class PreTrainedModel
    """
    for layer, heads in heads_to_prune.items():
      self.encoder.layer[layer].attention.prune_heads(heads)

  def forward(self, input_ids, input_ids_aa, input_ids_label, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, word_word_relation=None):
    # if attention_mask is None:
    #   attention_mask = torch.ones_like(input_ids) ## probably don't need this very much. if we pass in mask and token_type, which we always do for batch mode
    # if token_type_ids is None:
    #   token_type_ids = torch.zeros_like(input_ids)

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    if head_mask is not None:
      if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
      elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
      head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
      head_mask = [None] * self.config.num_hidden_layers

    ## need to split the @input_ids into AA side and label side, @input_ids_aa @input_ids_label
    embedding_output = self.embeddings(input_ids_aa, position_ids=position_ids, token_type_ids=token_type_ids)
    embedding_output_label = self.embeddings_label(input_ids_label, position_ids=None, token_type_ids=None)

    # concat into the original embedding
    embedding_output = torch.cat([embedding_output,embedding_output_label], dim=1) ## @embedding_output is batch x num_aa x dim so append @embedding_output_label to dim=1 (basically adding more words to @embedding_output)

    # @embedding_output is just some type of embedding, the @encoder will apply attention weights
    encoder_outputs = self.encoder(embedding_output,
                                   extended_attention_mask, ## must mask using the entire set of sequence + label input
                                   head_mask=head_mask,
																	 word_word_relation=word_word_relation)

    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertForTokenClsDistance (BertPreTrainedModel):

  def __init__(self, config):
    super(BertForTokenClsDistance, self).__init__(config)

    self.num_labels = 2 # config.num_labels ## for us, each output vector is "yes/no", so we should keep this at self.num_labels=2 to avoid any strange error later

    self.bert = BertModelDistance(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # self.apply(self.init_weights)
    # self.init_weights() # https://github.com/lonePatient/Bert-Multi-Label-Text-Classification/issues/19

    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

  def init_label_emb(self,pretrained_weight):
    self.bert.embeddings_label.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))
    self.bert.embeddings_label.word_embeddings.weight.requires_grad = False

  def forward(self, input_ids, input_ids_aa, input_ids_label, token_type_ids=None, attention_mask=None, labels=None,
        position_ids=None, head_mask=None, attention_mask_label=None, word_word_relation=None):

    ## !! add @attention_mask_label
    ## !! add @input_ids, @input_ids_aa. Label side is computed differently from amino acid side.

    outputs = self.bert(input_ids, input_ids_aa, input_ids_label, position_ids=position_ids, token_type_ids=token_type_ids,
              attention_mask=attention_mask, head_mask=head_mask, word_word_relation=word_word_relation)

    sequence_output = outputs[0] ## last layer.

    ## @sequence_output is something like batch x num_label x dim_out(should be 768)
    sequence_output = self.dropout(sequence_output)

    logits = self.classifier(sequence_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
    if labels is not None:
      loss_fct = CrossEntropyLoss()

      ## must extract only the label side (i.e. 2nd sentence)
      ## last layer outputs is batch_num x len_sent x dim
      ## we can restrict where the label start. and where it ends ??
      ## notice, @attention_mask is used to avoid padding in the @active_loss
      ## so we need to only create another @attention_mask_label to pay attention to labels only

      # Only keep active parts of the loss
      if attention_mask_label is not None: ## change @attention_mask --> @attention_mask_label
        active_loss = attention_mask_label.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1) # [active_loss] ## do not need to extract labels ?? we can pass in the exact true label
        loss = loss_fct(active_logits, active_labels)
      else:
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,active_logits,) + outputs

    return outputs  # (loss), scores, (hidden_states), (attentions)

