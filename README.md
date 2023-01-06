# Polysemia: Visualize Contextual Word Emebeddings

The script `main.py` is designed to accept a single English word, search the Gutenberg corpus for uses of that word in context, feed that context to BERT, and extract context-dependent feature embeddings for that word in each of its uses. The processed data will be saved in a compressed PyTorch file in a new folder called `cache/`. The user can then run `visualize.ipnyb` to preform principal component analysis on the emebeddings and create an interactive 3d visualization of the PCA plot. Different meanings of the word will show up as distinct clusters in the plot. 

`main.py` accepts two arguments: `--word` defines the word to search for, and `--numfiles` will limit the number of books searched for that word. Searching the whole corpus is quite expensive, and good plots can be obtained with only a few thousand books. Note that only words in the BERT vocabulary are accepted. The program will use CUDA for inference if it is available; running on CPU is not recommended for large sample sizes. 

This is a work in progress!

## Installation Instructions

This program uses the [Standardized Project Gutenberg Corpus](https://arxiv.org/abs/1812.08092) as a data source. To download it, navigate into the main directory of the project and run the following commands:

```sh
curl https://zenodo.org/record/2422561/files/SPGC-tokens-2018-07-18.zip?download=1 > tmp.zip
unzip tmp.zip
rm tmp.zip
```

I have included the metadata file which comes with the dataset in this repository due to its small size, but it can also be downloaded from their site with

```sh
curl https://zenodo.org/record/2422561/files/SPGC-metadata-2018-07-18.csv?download=1 > metadata.csv
```

If you want the repo to work out of the box, it is recommended to set up a virtual environment and install the required packages:
```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
The visualizations are designed to work with the integrated notebook viewer in VSCode. You may need to modify `pio.renderers.default` to get it to work in a different context.
