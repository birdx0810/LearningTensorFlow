# introduction
Word Vectors <=> Word Embeddings
- a set of language modeling techniques for mapping words to a vector of numbers (turns a text into numbers)
- a numeric vector represents a word
- comparitively sparse: more words == higher dimension

Key properties for Embeddings:
- Dimensionality Reduction: a more efficient representation
- Contextual Similarity: a more expressive representation
  - Syntax(syntactic): Grammatical structure
  - Semantics(Sentiment): Meaning of vocabulary

## Text Pre-processing
- removing tags (HTML, XML)
- removing accented characters (é)
- expanding contractions (don't, i'd)
- removing special characters (!@#$%^&\*)
- stemming and lemmatization
    - remove affixes
    - root word/stem
- removing stopwords (a, an, the, and)
- remove whitespace, lowercasing, spelling/grammar corrections etc.
- replace special tokens (digits to `[NUM]` token)
- [Example Code](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/Text%20Normalization%20Demo.ipynb)