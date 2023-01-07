# Polysemia: Visualize Contextual Word Emebeddings

This is a work in progress!

The script `main.py` is designed to accept a single English word, search the Gutenberg corpus for uses of that word in context, feed that context to BERT, and extract context-dependent feature embeddings for that word in each of its uses. The processed data will be saved in a compressed PyTorch file in a new folder called `cache/`. The user can then run `visualize.ipnyb` to preform principal component analysis on the emebeddings and create an interactive 3d visualization of the PCA plot. Different meanings of the word will show up as distinct clusters in the plot. 

`main.py` accepts two arguments: `--word` defines the word to search for, and `--numfiles` will limit the number of books searched for that word. Searching the whole corpus is quite expensive, and good plots can be obtained with only a few thousand books. Note that only words in the BERT vocabulary are accepted. The program will use CUDA for inference if it is available; running on CPU is not recommended for large sample sizes.

### Example Run

I ran the command 
```sh
python main.py --word right --numfiles 1000
```
As a test machine, I used a friend's computer with a Ryzen 3600XT 6-core CPU and an RTX 3090. Data extraction took 3 minutes, finding 46 thousand examples. Inference on those examples took another 7 minutes (inference with a CPU would take around 9 hours!). The generated plot is shown below (right)
<p align="center">
    <img src="https://github.com/a-g-moore/polysemia/blob/master/example.png?raw=true" width="40%">
    <img src="https://github.com/a-g-moore/polysemia/blob/master/example_colored.png?raw=true" width="48%">
</p>
The word 'right' has a variety of related meanings, each of which is clustered in its own blob on the chart. The tooltip shows the metadata for a point in the rightmost blob, which corresponds to 'right' meaning 'moral or legal entitlement'. On the right, we have computed the PCA with 10 components and then applied the OPTICS clustering algorithm, which attempts to automatically color the datapoints by grouping, thus distinguishing the definition. The vast majority of points are discarded as noise; here, only the core points are plotted. OPTICS was chosen as the clustering algorithm because it is able (unlike K-means or EM clustering) to cluster points without knowing the number of clusters in advance, which is vital for this application. It was chosen over DBSCAN, the other viable algorithm, because due to differences in the frequency of different uses of words, clusters may have rather different densities, and tuning hyperparameters may not generalize well between different words. The clustering algorithm is allowed to use 10 PCA dimensions, which significantly improves its preformance over using only the 3 plotted principal dimensions---but we do not necessarily want to give the clustering algorithm too many dimensions, since metric-based algorithms tend to suffer from the curse of dimensionality. OPTICS is slow (takes around a minutes to cluster these 46 thousand points), but the slowdown is perfectly acceptable due to the importance of high accuracy. 

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
