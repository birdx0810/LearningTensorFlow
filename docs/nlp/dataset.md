# NLP Datasets

## bABI
- [bAbI](https://research.fb.com/downloads/babi/)(ICLR 2015): A set of prerequisite toy tasks for NLP

(Unofficial) Leaderboard: [Link](https://paperswithcode.com/sota/question-answering-on-babi)
GitHub: [Link](https://github.com/facebook/bAbI-tasks)
- The file format for each task is as follows:
```
ID text
ID text
ID text
ID question[tab]answer[tab]supporting fact IDS.
...
```
- For example:
```
...
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
...
```

- $\text{ID} \to 1$ == new story
- qa2_train.txt
    - Factoid QA with two supporting facts
    - Question asks **where is object**?

## Social Networks
- Twitter
  - Official API: [Documentation](https://developer.twitter.com/en/docs)
  - 3rd Party Python Wrappers:
    - [Tweepy](https://tweepy.readthedocs.io/en/latest/)
    - [Python-Twitter](https://github.com/bear/python-twitter)
- Reddit
  - Official API: [Documentation](https://www.reddit.com/dev/api)
  - 3rd Party Python Wrappers:
    - [PRAW](https://praw.readthedocs.io/en/latest/)

## Conversational Models
### English
<!-- - A Survey of Available Corpora for Building Data-Driven Dialogue Systems]([Serban et al., 2015](https://breakend.github.io/DialogDatasets/)) -->
<!-- A Repository of Conversational Datasets: [GitHub](https://github.com/PolyAI-LDN/conversational-datasets) -->
<!-- - [Twitter](https://www.kaggle.com/thoughtvector/customer-support-on-twitter)(Kaggle 2017): A large corpus of modern English (mostly) conversations between consumers and customer support agents on Twitter -->
- [Relational Strategies in Customer Service (RSiCS) Dataset](https://nextit-public.s3-us-west-2.amazonaws.com/rsics.html) ([AAAI 2020](https://nextit-public.s3-us-west-2.amazonaws.com/AAAI-BeaverI.2299.pdf))
- [Coached Conversational Preference Elicitation](https://research.google/tools/datasets/coached-conversational-preference-elicitation/) (2019)
- [Taskmaster-1](https://research.google/tools/datasets/taskmaster-1/) & [Taskmaster-2](https://research.google/tools/datasets/taskmaster-2/) (2019)
- Restoration-200K Dataset (EMNLP 2019): 200K Multi-turn Open-domain conversations with six utterances each. [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/Restoration-200K.zip)
- Multi-relation Question Answering Dataset (COLING 2018): Multi-relation question answering over knowledge bases. [Download](http://coai.cs.tsinghua.edu.cn/hml/media/files/PathQuestion.zip)
- Commonsense Conversation (IJCAI 2018): One-turn post-response pair with corresponding commonsense knowledge graph [Download](http://coai.cs.tsinghua.edu.cn/file/commonsense_conversation_dataset.tar.gz)
- [**ConvAI2**](http://convai.io/data/) Competition (NIPS 2018)
- [DailyDialog](http://yanran.li/dailydialog) (IJCNLP 2017): Human-written and less noisy that reflects daily communication of humans
- [OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php) (LREC 2016): A new collection of translated movie subtitles from http://www.opensubtitles.org/
- [ChangeMyView](https://chenhaot.com/pages/changemyview.html) (WWW 2016): Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions
- [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator) (SIGDIAL 2015): Unstructured Multi-turn Dialogue System
- [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) (2011): Contains a large collection of fictional conversations from raw movie scripts

### Chinese
- [CrossWOZ](https://github.com/thu-coai/CrossWOZ) (TACL 2020): A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset
- [KdConv](https://github.com/thu-coai/KdConv) (ACM 2020): A Chinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation
- Grayscale Dataset for Dialogue (2020): [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/grayscale_data_release.zip)
- Retrieval Generation Chat (EMNLP 2019): [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/Retrieval_Generation_Chat.zip)
- Chinese Dialogue Sentence Function Dataset (ACL 2019): [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/dialog-acts.tar.gz)
- Weibo Conversation Dataset (AAAI 2019): [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/weibo_utf8.zip)
- Dialogue Question Generation Dataset (ACL 2018): [Download](http://coai.cs.tsinghua.edu.cn/file/QGdata.zip)
- Emotional Conversation Dataset (AAAI 2018): [Link](http://coai.cs.tsinghua.edu.cn/hml/challenge2017/)
- [Douban Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection) (ACL 2017)
- [Insurance QA](https://github.com/chatopera/insuranceqa-corpus-zh) (chatopera)

### Style
- Gender-Specific Chat Dataset (2020): [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/Stylistic_Dataset.zip)
- Chinese Literal Style Dataset (EMNLP 2019): [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/poem_corpus.zip)
- Personality Assignment Dataset (IJCAI 2018): [Download](http://coai.cs.tsinghua.edu.cn/file/ijcai_data.zip)

## Summarization
- [Multi-News](https://github.com/Alex-Fabbri/Multi-News): Large-Scale Multi-Document Summarization Dataset
- [CNN-DailyMail](https://github.com/abisee/cnn-dailymail): Dataset for Pointer-Generator by Abigail See
- [GigaWord](https://drive.google.com/open?id=1eNUzf015MhbjOZBpRQOfEqjdPwNz9iiS): Dataset for Pointer-Generator by Abigail See
