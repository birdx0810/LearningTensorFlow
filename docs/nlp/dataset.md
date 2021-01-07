# NLP Datasets and Benchmarks
## Benchmarks
### General Language Understanding Evaluation (GLUE)
- Website: [Link](https://gluebenchmark.com/)
- Natural Language Understanding
  - CoLA (grammatical acceptability): [Link](https://nyu-mll.github.io/CoLA/)
  - SST (sentiment analysis): [Link](https://nlp.stanford.edu/sentiment/index.html)
  - MRPC (sentence similarity): [Link](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
  - STS (sentence similarity): [Link](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)
  - QQP (sentence similarity): [Link](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
  - MNLI (natural language inference): [Link](https://cims.nyu.edu/~sbowman/multinli/) 
  - QNLI (natural language inference): [Link](https://rajpurkar.github.io/SQuAD-explorer/)
  - RTE (natural language inference): [Link](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment)
  - WNLI (natural language inference): [Link](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)
  - Diagnostics: [Link](https://gluebenchmark.com/diagnostics)
- Variants: SuperGLUE, CLUE, BLUE

### Stanford Question Answering Dataset (SQuAD)
- Website: [Link](https://rajpurkar.github.io/SQuAD-explorer/)
- Reading Comprehension (QA)
  - Find the answer reference (span) from the document
  - 2.0 also considers unanswerable questions

## Datasets
### A set of prerequisite toy tasks for NLP (bABI)
- Website: [Link](https://research.fb.com/downloads/babi/)(ICLR 2015)

The file format for each task is as follows:
```
ID text
ID text
ID text
ID question[tab]answer[tab]supporting fact IDS.
...
```

For example:
```
...
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
...
```

### Summarization
- [Multi-News](https://github.com/Alex-Fabbri/Multi-News): Large-Scale Multi-Document Summarization Dataset
- [CNN-DailyMail](https://github.com/abisee/cnn-dailymail): Dataset for Pointer-Generator by Abigail See
- [GigaWord](https://drive.google.com/open?id=1eNUzf015MhbjOZBpRQOfEqjdPwNz9iiS): Dataset for Pointer-Generator by Abigail See
