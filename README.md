# RPMOG

The source code for paper "Retrospective and Prospective Mixture-of-Generators for Task-oriented Dialogue Response Generation"

# Requirements
Python 3 with pip

# Quick start
In repo directory:

## Install the required packages
- Using Conda:
```console
cd multiwoz-mdrg
conda create --name multiwoz python=3.7 anaconda
source activate multiwoz
conda install --file requirements.txt 
conda install pytorch torchvision -c pytorch
```  

## Preprocessing
To download and pre-process the data run:

```python multiwoz/Create_delex_data.py```

## For debugging
To debug train.py, you can add the following parameteres to save time
--debug=True --emb_size=5 --hid_size_dec=5 --hid_size_enc=5 --hid_size_pol=5 --max_epochs=2

To debug test.py, the parameters are:
--debug=True --no_models=2 --beam_width=2

## Training
To train the model run:

```python train.py [--args=value]```

Some of these args include:

```
// hyperparamters for model learning
--max_epochs        : numbers of epochs
--batch_size        : numbers of turns per batch
--lr_rate           : initial learning rate
--clip              : size of clipping
--l2_norm           : l2-regularization weight
--dropout           : dropout rate
--optim             : optimization method

// network structure
--emb_size          : word vectors emedding size
--use_attn          : whether to use attention
--hid_size_enc      : size of RNN hidden cell
--hid_size_pol      : size of policy hidden output
--hid_size_dec      : size of RNN hidden cell
--cell_type         : specify RNN type
```

## Testing
To evaluate the run:

```python test.py [--args=value]```

## Hyperparamters
```
// hyperparamters for model learning
--max_epochs        : 20
--batch_size        : 64
--lr_rate           : 0.005
--clip              : 5.0
--l2_norm           : 0.00001
--dropout           : 0.0
--optim             : Adam

// network structure
--emb_size          : 50
--use_attn          : True
--hid_size_enc      : 150
--hid_size_pol      : 150
--hid_size_dec      : 150
--cell_type         : gru
```
