# Handwriting with Pytorch

![alignment_01](https://raw.githubusercontent.com/anomam/handwriting-pytorch/main/log/plots/example_generated_01.png)

This is my implementation attempt of [this amazing paper](https://arxiv.org/abs/1308.0850) from Alex Graves, using Pytorch.  
So far I've only implemented the unconditional handwriting generation part, but I'll add the conditional generation one as soon as I can.

## How to

### Install dependencies

The dependencies can be installed using:

```
pip install -r requirements.txt
```

You might need to adjust the pytorch version to your local setup though (CUDA version, ...): see this Pytorch installation [matrix](https://pytorch.org/get-started/locally/) for help.

### See all the available commands

Simply use:

```
python cli.py --help
```

### Train the LSTM network

#### Data download

The raw data needs to be downloaded from the [official database](http://www.iam.unibe.ch/fki/databases/iam-on-line-handwriting-database) into this folder: `data/data_raw/`

#### Prepare data

You can then prepare the data by running:

```
python cli.py prepare-data
```

It will convert the raw data into numpy arrays that can be easily loaded from the `data/data_np_offsets/` directory.

#### Training

You can launch the training by running:

```
python cli.py train-generator
```

A number of parameters can be passed to this command; use the following for more details:

```
python cli.py train-generator --help
```

Training for 20 epochs led to good enough results. As an point of reference: 1 epoch takes around 20s on my GPU (using the default command parameters).  
The saved model parameters as well as the in-training generated examples will be saved into the `data/log/` folder.

### Generate examples

Unconditional generation of examples can be done using:

```
python cli.py generate
```

The generated results will be saved into `data/log/plots/`

## Unconditional generation examples

Generated examples with no conditioning on the text characters:

![no_alignment](https://raw.githubusercontent.com/anomam/handwriting-pytorch/main/log/plots/example_generated_no_align.png)

![alignment_01](https://raw.githubusercontent.com/anomam/handwriting-pytorch/main/log/plots/example_generated_01.png)

## Resources

I would like to give a big acknowledgment to these awesome repositories that helped me a lot understand how to parse & transform the data, and the intricacies of the RNN architecture implementation with the negative log-likelihood loss function:

- [handwriting-synthesis](https://github.com/swechhachoudhary/Handwriting-synthesis/) by [swechhachoudhary](https://github.com/swechhachoudhary)
- [handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis/) by [sjvasquez](https://github.com/sjvasquez)
- [write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow/) by [hardmaru](https://github.com/hardmaru)
