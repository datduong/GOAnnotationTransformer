


## create a new BERT where the emb takes in Kmer sequence ? 

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer


class BertForTokenClassification1hot (BertPreTrainedModel):

  ## !! we change this to do 1-hot prediction 
  ## take in K labels so we have vector of 1-hot length K
  ## for each label, we get a vector output from BERT, then we predict 0/1 

  r"""
    **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
      Labels for computing the token classification loss.
      Indices should be in ``[0, ..., config.num_labels]``.

  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
    **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
      Classification loss.
    **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
      Classification scores (before SoftMax).
    **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
      list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
      of shape ``(batch_size, sequence_length, hidden_size)``:
      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    **attentions**: (`optional`, returned when ``config.output_attentions=True``)
      list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

  Examples::

    >>> config = BertConfig.from_pretrained('bert-base-uncased')
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>>
    >>> model = BertForTokenClassification(config)
    >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    >>> labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
    >>> outputs = model(input_ids, labels=labels)
    >>> loss, scores = outputs[:2]

  """
  def __init__(self, config):
    super(BertForTokenClassification1hot, self).__init__(config)
    self.num_labels = 2 # config.num_labels ## for us, each output vector is "yes/no", so we should keep this at self.num_labels=2 to avoid any strange error later

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    # self.apply(self.init_weights)
    self.init_weights() # https://github.com/lonePatient/Bert-Multi-Label-Text-Classification/issues/19

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
        position_ids=None, head_mask=None, attention_mask_label=None):
    
    ## !! add @attention_mask_label

    outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
              attention_mask=attention_mask, head_mask=head_mask)
    
    sequence_output = outputs[0] ## last layer. 
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

