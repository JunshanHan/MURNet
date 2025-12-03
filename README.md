# Development, Evaluation and Application of a Multi-Representation Fusion Model for Accurate Prediction of Per- and Polyfluoralkyl Substance(PFAS) Binding to Plasma Proteins

This repository is the official implementation of **MURNet**, which is model proposed in a paper: **Development, Evaluation and Application of a Multi-Representation Fusion Model for Accurate Prediction of Per- and Polyfluoralkyl Substance(PFAS) Binding to Plasma Proteins**.

MURNet is a valuable method for identifying and evaluating PFAS in the environment that can bind to plasma proteins.

![An illustration of MURNet and its workflow.](/imgs/2025-12-03/mHe6wa0PXSde6qpn.png)
 
 ## # Requirements
 To run our code, please install dependency packages.
~~~
python        3.7
torch         1.13.1
rdkit         2018.09.3
numpy         1.20.3
gensim        4.2.0
nltk          3.4.5
owl2vec-star  0.2.1
Owlready2     0.37
torch-scatter 2.0.9
~~~

## Overview
This project mainly contains the following parts.
~~~
├── baseline               #six contrast algorithms
│   ├── DNN.py
│   ├── GCN.py
│   ├── LGB.py
│   ├── RF.py
│   ├── SVM.py
│   ├── two layer MLP.py   #the prediction model
│   └── XGB.py
├── data                   #the data used by MURNet
│   ├── 65-HSA.csv
│   ├── OECD.csv
│   └── PFAS_1.csv
├── Exploration of Influencing Factors
│   ├── Calculation of F atom number and heavy atom number.py
│   ├── Chain length - label - box plot.py
│   ├── Density curve drawing.py
│   ├── F atomic number -XlogP scatter plot drawing.py
│   └── smiles to XLogP.py
├── The alignment and uniformity of the characterization
│   ├── embedding-visualization.py
│   ├── functional group-name+emb-extract.py
│   ├── KDE.py
│   ├── representation-visualization.py
│   └── TOP5scaffolds.py
~~~

## Making predictions
~~~
two layer MLP.py
~~~

## Acknowledgements
Thanks for the following released code bases:
> [chemprop](https://github.com/chemprop/chemprop), [torchlight](https://github.com/RamonYeung/torchlight), [RDKit](https://github.com/rdkit/rdkit), [KCL](https://github.com/ZJU-Fangyin/KCL)

## About
Should you have any questions, please feel free to contact Miss Junshan Han at [hanjunshan01@163.com](mailto:hanjunshan01@163.com).
