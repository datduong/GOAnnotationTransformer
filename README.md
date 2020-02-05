# GOAT: GO Annotation with the Transformer model 

### [This is our paper.](https://www.biorxiv.org/content/10.1101/2020.01.31.929604v1)

## Pre-trained model, and training your own model

We adapt the Transformer neural network model to predict GO labels for protein sequences. You can download our trained models are here. 

You can also train your own model, see example shell script here. Your input must match the input here. 

**GOAT supports 4 training options:** (see example shell script to choose each option)
1. Base Transformer
2. Domain data (like motifs, compositional bias, etc.)
3. External protein data (like 3D structure, protein-protein interaction network)
4. Any combination of the above. 



