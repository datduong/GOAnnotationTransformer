

### Train Transformer model to predict GO labels of protein sequences.

We discuss some key points about training this model. This code is built from [Pytorch Transformer](https://github.com/huggingface/transformers). Our model uses only **1 head and 12 layer Transformer**, but you can always expand the numbers of parameters. You can edit our Transformer configuration here. 

There is a vocab.txt that was redesign to recognize only amino acids. We did not pre-train the sequences on any Language Model, because this step would require a lot of samples (in the millions). 

The demo script shows:
1. How to train the model.
2. How to test the trained model. 
3. How to perform zeroshot on unseen labels (unimpressive accuracy). 
4. How to extract GO label vectors from layer 12 of Transformer. 




