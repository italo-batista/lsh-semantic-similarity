

LSH (Locality Sensitive Hashing) is primarily used to find, given a large set of documents, the near-duplicates among them.
It can use hamming distance, jaccard coefficient, edit distance or other distance notion. 
The main idea is to use hash collisions to capture objects similarities.

You can read thw following tutorials if you want to understand more about it:

- [Ravi Kumar's work](https://users.soe.ucsc.edu/~niejiazhong/slides/kumar.pdf)
- [Matti Lyra's work](https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html)
- [Insideops](https://insideops.wordpress.com/2015/07/30/similarity-search-and-hashing-for-text-documents/)

Although LSH is more to duplicated documents than to semantic similar ones, in this approach I make an effort to use LSH 
to calculate semantic similarity among texts. For that, the algorithm extracts, using TFIDF, the text's main tokens 
(or you can pre-calculate them and pass as param). Also, in this approach I use MinHash (which uses Jaccard similarity) as the 
Similarity function.

**The overall aim is to reduce the number of comparisons needed to find similar items.**
The hash collisions come in handy here as similar documents have a high probability of having the same hash value. 
The probability of a hash collision for a minhash is exactly the Jaccard similarity of two sets.

See [this tutorial](tutorial.ipynb) to see how use this LSH!

Run as following to install dependencies:

```
  python3 -m pip install -r requirements.txt
```
