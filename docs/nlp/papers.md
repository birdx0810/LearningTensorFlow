# Must Read NLP Papers

## Word Embeddings
### Pre-Word2Vec
1. A unified architecture for natural language processing: deep neural networks with multitask learning. Ronan Collobert, Jason Weston. ICML 2008. [paper](https://openreview.net/forum?id=HJWhcjWOWS)
2. Word Representations: A Simple and General Method for Semi-Supervised Learning. Joseph Turian, Lev-Arie Ratinov, Yoshua Bengio. NAACL 2010. [paper](https://www.aclweb.org/anthology/P10-1040/)
3. Linguistic Regularities in Continuous Space Word Representations. Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig. NAACL 2013. [paper](https://www.aclweb.org/anthology/N13-1090/).

### Post-Word2Vec
1. CBOW & Skip-Gram: Efficient Estimation of Word Representations in Vector Space. Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. ICLR 2013.
2. Hierarchical Softmax & Negative Sampling: Distributed Representations of Words and Phrasesand their Compositionality. Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. NIPS 2013.
3. GloVe: Global Vectors for Word Representation. Jeffrey Pennington, Richard Socher, Christopher D. Manning. 
4. fastText: Enriching Word Vectors with Subword Information. P. Bojanowski, E. Grave, A. Joulin, T. Mikolov.
5. ELMo: Deep contextualized word representations. Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer. NAACL 2018.
6. ULMFiT: Universal Language Model Fine-tuning for Text Classification. Jeremy Howard and Sebastian Ruder. ACL 2018.
7. MultiFiT: Efficient Multi-lingual Language Model Fine-tuning. Julian Eisenschlos, Sebastian Ruder, Piotr Czapla, Marcin Kardas, Sylvain Gugger, Jeremy Howard. EMNLP 2019.
8. UniLM: Unified Language Model Pre-training for Natural Language Understanding and Generation. Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon. Preprint. 

## Tokenizers
0. WordPiece: Japanese and Korean Voice Search. Mike Schuster, Kaisuke Nakajima. ICASS 2012. [tech-report](https://research.google/pubs/pub37842/)
1. Byte Pair Encoding: Neural Machine Translation of Rare Words with Subword Units. Rico Sennrich, Barry Haddow, Alexandra Birch. ACL 2016. [paper](https://arxiv.org/abs/1508.07909)
2. Byte-to-Span: Multilingual Language Processing From Bytes. Dan Gillick, Cliff Brunk, Oriol Vinyals, Amarnag Subramanya. NAACL 2016. [paper](https://arxiv.org/abs/1512.00103)
3. SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. Taku Kudo, John Richardson. EMNLP 2018. [paper](https://arxiv.org/abs/1808.06226)
4. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. Taku Kudo. ACL 2018. [paper](https://www.aclweb.org/anthology/P18-1007/)

## Language Models
1. Class-based n-gram models of natural language. Peter Brown, Peter DeSouza, Robert Mercer, Vincent Della Pietra, Jenifer Lai. Computer Linguistics 1992. [paper](https://dl.acm.org/doi/abs/10.5555/176313.176316)
2. Neural Probabilistic Language Model. Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. JMLR 2003. [paper](https://www.jmlr.org/papers/v3/bengio03a.html) 
3. Recurrent Neural Network Based Language Model. Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, Sanjeev Khudanpur. INTERSPEECH 2010. [paper](https://www.isca-speech.org/archive/interspeech_2010/i10_1045.html)
4. Generating Sequences with Recurrent Neural Networks. Alex Graves. [paper](https://arxiv.org/abs/1308.0850) 

## Decoding Algorithms
1. The Curious Case of Neural Text Degeneration. Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. ICLR 2020. [paper](https://arxiv.org/abs/1904.09751)
2. Trading Off Diversity and Quality in Natural Language Generation. Hugh Zhang, Daniel Duckworth, Daphne Ippolito, Arvind Neelakantan. Preprint. [paper](https://arxiv.org/abs/2004.10450)

## Sequence To Sequence
1. Sequence to Sequence Learning with Neural Networks. Ilya Sutskever, Oriol Vinyals, and Quoc VLe. NIPS 2014
2. Semi-supervised Sequence Learning. Andrew M. Dai, Quoc V. Le. NIPS 2015.
3. Neural Machine Translation by Jointly Learning to Align and Translate. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. ICLR 2015
4. Unsupervised Pretraining for Sequence to Sequence Learning. Prajit Ramachandran, Peter J. Liu, Quoc V. Le. EMNLP 2017.
5. Attention is All You Need. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan NGomez, Lukasz Kaiser, and Illia Polosukhin. NIPS 2017

## Post-Transformers
1. GPT: Improving Language Understanding by Generative Pre-Training. Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. Preprint.
2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. NAACL 2019.
3. GPT2: Language Models are Unsupervised Multitask Learners. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. Preprint.
4. MT-DNN: Multi-Task Deep Neural Networks for Natural Language Understanding. Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao. ACL 2019.
5. RoBERTa: A Robustly Optimized BERT Pretraining Approach. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. Preprint. 
6. SpanBERT: Improving Pre-training by Representing and Predicting Spans. Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy. Preprint.
7. Chinese-BERT-wwm: Pre-Training with Whole Word Masking for Chinese BERT. Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, Guoping Hu. Preprint.
8. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. ICLR 2020.
9. XLM: Cross-lingual Language Model Pretraining. Guillaume Lample, Alexis Conneau. NeurIPS 2019.
10. XLNet: Generalized Autoregressive Pretraining for Language Understanding. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. NeurIPS 2019.
11. T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. Preprint.
12. MASS: Masked Sequence to Sequence Pre-training for Language Generation. Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu. ICML 2019.
13. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer. ACL 2020.
14. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning. ICLR 2020.

## Sesame Street
1. GROVER: Defending Against Neural Fake News. Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, Yejin Choi. [blog](https://rowanzellers.com/grover/)
2. ERNIE: Enhanced Language Representation with Informative Entities. Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun and Qun Liu. ACL 2019. [paper](https://arxiv.org/abs/1905.07129)
3. ERNIE: Enhanced Representation through Knowledge Integration. Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian and Hua Wu. Preprint. [paper](https://arxiv.org/abs/1904.09223)
4. ERNIE 2.0: A Continual Pre-training Framework for Language Understanding. Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, Haifeng Wang. AAAI 2020. [paper](https://arxiv.org/abs/1907.12412)
5. ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation. Preprint. [paper](https://arxiv.org/abs/2001.11314)
6. ERNIE-Joint: An ERNIE-Based Joint Model for Chinese Named Entity Recognition. Yu Wang, Yining Sun, Zuchang Ma, Lisheng Gao,and Yang Xu. [paper](https://www.mdpi.com/2076-3417/10/16/5711) 
7. ERNIESage: ERNIE SAmple aggreGatE. [code](https://github.com/PaddlePaddle/PGL/tree/master/examples/erniesage)
6. KERMIT: Complementing Transformer Architectures with Encoders of Explicit Syntactic Interpretations. Fabio Massimo Zanzotto, Andrea Santilli, Leonardo Ranaldi, Dario Onorati, Pierfrancesco Tommasino, Francesca Fallucchi. EMNLP 2020. [paper](https://www.aclweb.org/anthology/2020.emnlp-main.18/)
7. KERMIT: Generative Insertion-Based Modeling for Sequences. William Chan, Nikita Kitaev, Kelvin Guu, Mitchell Stern, Jakob Uszkoreit. Preprint. [paper](https://arxiv.org/abs/1906.01604)
8. Big Bird: Big Bidirectional Insertion Representations for Documents. Lala Li, William Chan. Preprint. [paper](https://arxiv.org/abs/1910.13034)
9. Big Bird: Transformers for Longer Sequences. Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed. Preprint [paper](https://arxiv.org/abs/2007.14062)

## VAE
1. Generating Sentences from a Continuous Space. Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, Samy Bengio. CONLL 2016. [paper](https://arxiv.org/abs/1511.06349)
2. Toward Controlled Generation of Text. Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric P. Xing. PMLR 2017. [paper](https://arxiv.org/abs/1703.00955)
3. Topic-Guided Variational Autoencoders for Text Generation. Wenlin Wang, Zhe Gan, Hongteng Xu, Ruiyi Zhang, Guoyin Wang, Dinghan Shen, Changyou Chen, Lawrence Carin. NAACL 2019. [paper](https://arxiv.org/abs/1903.07137)

## GAN
1. Generating Text via Adversarial Training. Yizhe Zhang and Zhe Gan. NIPS GAN Workshop 2016. [paper](https://sites.google.com/site/nips2016adversarial/WAT16_paper_20.pdf)
2. Adversarial Generation of Natural Language. Sai Rajeswar, Sandeep Subramanian, Francis Dutil, Christopher Pal, Aaron Courville. RepL4NLP 2017 Workshop. [paper](https://arxiv.org/abs/1705.10929)
2. SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu. AAAI 2017. [paper](https://arxiv.org/abs/1609.05473) 
3. TextGAN: Adversarial Feature Matching for Text Generation. Yizhe Zhang, Zhe Gan, Kai Fan, Zhi Chen, Ricardo Henao, Dinghan Shen, Lawrence Carin. ICML 2017. [paper](https://arxiv.org/pdf/1706.03850.pdf)
4. Adversarial Learning for Neural Dialogue Generation. Jiwei Li, Will Monroe, Tianlin Shi, Sébastien Jean, Alan Ritter, Dan Jurafsky. EMNLP 2017. [paper](https://arxiv.org/abs/1701.06547)
5. LeakGAN: Long Text Generation via Adversarial Training with Leaked Information. Jiaxian Guo, Sidi Lu, Han Cai, Weinan Zhang, Yong Yu, Jun Wang. AAAI 2018. [paper](https://arxiv.org/abs/1709.08624) 
6. MaskGAN: Better Text Generation via Filling in the______. William Fedus, Ian Goodfellow, Andrew M. Dai. ICLR 2018. [paper](https://arxiv.org/abs/1801.07736)
7. RelGAN: Relational Generative Adversarial Networks for Text Generation. Weili Nie, Nina Narodytska, Ankit Patel. ICLR 2019. [paper](https://openreview.net/forum?id=rJedV3R5tm)

## Knowledge Integration
1. A Neural Knowledge Language Model. Sungjin Ahn, Heeyoul Choi, Tanel Pärnamaa, Yoshua Bengio. [paper](https://arxiv.org/abs/1608.00318)

## Topic Specific
### Machine Translation
1. Neural Machine Translation by Jointly Learning to Align and Translate. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. ICLR 2015. [paper](https://arxiv.org/abs/1409.0473)

### Summarization
1. Pointer-Generator Networks. Abigail See, Peter J. Liu, Christopher D. Manning. ACL 2017. [paper](https://arxiv.org/abs/1704.04368)

### Question Answering
1. End-to-end Memory Networks. Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus. NIPS 2015. [paper](https://arxiv.org/abs/1503.08895)

### Conversational Models
1. A Neural Conversational Model. Oriol Vinyals, Quoc V. Le. [paper](https://arxiv.org/abs/1506.05869)

### Information Retrieval
1. Representation Learning Using Multi-Task Deep Neural Networks for Semantic Classification and Information Retrieval. Xiaodong Liu, Jianfeng Gao, Xiaodong He, Li Deng, Kevin Duh, Ye-yi Wang. NAACL 2015. [paper](https://www.aclweb.org/anthology/N15-1092/)

## Other
1. Natural Language Processing (almost) from Scratch. Ronan Collobert, Jason Weston, Leon Bottou, Michael Karlen, Koray Kavukcuoglu, Pavel Kuksa. JMLR 2011. [paper](https://arxiv.org/abs/1103.0398)
2. Embed, encode, attend, predict: The new deep learning formula for state-of-the-art NLP models. Matthew Honnibal. [blog](https://explosion.ai/blog/deep-learning-formula-nlp)