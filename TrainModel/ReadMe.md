

### Train Transformer model to predict GO labels of protein sequences.

We discuss some key points about training this model. This code is built from [Pytorch Transformer](https://github.com/huggingface/transformers). Our model uses only **1 head and 12 layer Transformer**, but you can always expand the numbers of parameters. You can edit [our Transformer configuration](https://drive.google.com/drive/folders/1MfjpaZ4Mg0L6PovPzfjAlB_ny1zYFFNm?usp=sharing) to increase the number of parameters. 

There is a vocab.txt that was redesign to recognize only amino acids. We did not pre-train the sequences on any Language Model, because this step would require a lot of samples (in the millions). 

We trained our models on the DeepGO datasets which was used as a baseline. **[Download the train/dev/test data here.](https://drive.google.com/drive/folders/1xwLnypz6JRUoQkbfdscG-NyusECVzQ7t?usp=sharing)**

Our **[demo script](https://github.com/datduong/GOAnnotationTransformer/tree/master/TrainModel/DemoScript)** shows:
1. How to specify a model (for example, base model v.s. model with protein network data)
2. How to train model.
3. How to test trained model. 
4. How to perform zeroshot on unseen labels (unimpressive accuracy). 
5. How to extract GO label vectors from layer 12 of Transformer. 

Please rename the directory paths as you go through the entire demo script. We were able to train the all model options for any datasets on one single Gtx 1080Ti having 11GB mem. Training on MF ontology takes about 12 hrs, on CC about 12 hrs, on BP about 16 hrs. 



