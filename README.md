# KGETCDA

This repo is the PyTorch implementation of "KGETCDA: an efficient representation learning framework based on knowledge graph encoder from transformer for predicting circRNA-disease associations"

## Introduction
**KGETCDA** (**K**nowledge **G**raph **E**ncoder from **T**ransformer for predicting **C**ircRNA-**D**isease **A**ssociations) is an efficient knowledge-based representation learning framework. Built upon knowledge graph and Transformer, KGETCDA can effectively produce high-quality embeddings with accurately captured low-order and high-order interaction information.

**KGETCDA** consistently achieves remarkable performance on three datasets (**Dataset1**---a small dataset focusing on non-cancer used in previous works; **Dataset2**---a larger heterogeneous dataset constructed by us; **Dataset3**---a small dataset focusing on cancer used in previous works)

We also provide a user-friendly interactive web-based platform (named **HNRBase**), which is publicly available at [http://lab-fly.site/KGETCDA](http://146.56.237.198:8012/).

## Requirements

The code has been tested running under Python 3.9.15. The required packages are as follows:
  * numpy == 1.24.1
  * pandas == 1.5.2
  * torch == 1.11.0

The expected structure of files is:

```
 ── KGETCDA
    ├── datasets
    │   ├── Dataset1
    │   ├── Dataset2
    │   └── Dataset3
    ├── data_loader
    │   └── loader_KGETCDA.py
    ├── models
    │   └── KGETCDA.py    
    ├── parsers
    │   └── parser_KGETCDA.py    
    ├── utils
    │   ├── helper_init.py
    │   ├── helper_log.py
    │   ├── helper_metrics.py
    │   └── helper_model.py
    ├── main.py
    └── test.py
```

## Dataset
We use three datasets here, and provide 4 entities (**circRNA**, **miRNA**, **lncRNA**, **disease**) and 5 relations (**circRNA-disease**, **miRNA-disease**, **lncRNA-disease**, **circRNA-miRNA**, **miRNA-lncRNA**) file and all pairs. The summary information is listed as follows:

| Dataset | circ-dis | mir-dis | lnc-dis | circ-mir | mir-lnc | total |
|:---:|:---|---:|---:|---:|---:|---:|
|Dataset1| 346 | 106 | 527 | 146 | 202 | 1327 |
|Dataset2| 1399 | 10154 | 3280 | 1129 | 9506 | 25468 |
|Dataset3| 647 | 732 | 1066 | 756 | 308 | 3509 |

* `entity.txt`
  * All entities file.
  * Each line is an entity with its ID: (`name` and `ID`).

* `relation.txt`
  * All relations file.
  * Each line is a relation with its ID: (`name` and `ID`).

* `pair_circ_dis.txt`
  * All circRNA-disease positve associations file.
  * Each line is a circRNA name with its positive interactions with diseases: (`name` and `name`).

* `pair_mir_dis.txt`
  * All miRNA-disease positve associations file.
  * Each line is a miRNA name with its positive interactions with diseases: (`name` and `name`).

* `pair_lnc_dis.txt`
  * All lncRNA-disease positve associations file.
  * Each line is a lncRNA name with its positive interactions with diseases: (`name` and `name`).

* `pair_circ_mir.txt`
  * All circRNA-miRNA positve associations file.
  * Each line is a circRNA name with its positive interactions with miRNAs: (`name` and `name`).

* `pair_mir_lnc.txt`
  * All miRNA-lncRNA positve associations file.
  * Each line is a miRNA name with its positive interactions with lncRNAs: (`name` and `name`).


## Usage
You can directly run the above model KGETCDA. 

We also recommend users use our KGETCDA Webserver ([http://lab-fly.site/KGETCDA](http://146.56.237.198:8012/)), which is user-friendly and easy to use. Consisting of 4 core functions (intelligent search and browse, model prediction, information visualization, and advanced interaction), our web-based platform enables novel visualization, accessible resources and user-friendly interaction. Everyone could upload or typein the candidate circRNAs or diseases of interest in our web without further installation, our backend server will calculate and give the prediction results to the user. Users can also choose to download the predict csv file results.

## Compared methods and related papers.
In this paper, we compare our model with 8 SOTAs including: GMNN2CD, KGANCDA, RNMFLP, AE-RF, DMFCDA, CD-LNLP, RWR, KATZHCDA, which are compared under the same experiment settings. The parameters of the other 8 models maintain consistency with their original papers.

* GMNN2CD (2022)
    * Proposed in [GMNN2CD: identification of circRNA–disease associations based on variational inference and graph Markov neural networks](https://academic.oup.com/bioinformatics/article/38/8/2246/6528308), Bioinformatics 2022.

* KGANCDA (2022)
    * Proposed in [KGANCDA: predicting circRNA-disease associations based on knowledge graph attention network](https://academic.oup.com/bib/article-abstract/23/1/bbab494/6447436?redirectedFrom=fulltext&login=false), Briefings in bioinformatics 2022.

* RNMFLP (2022)
    * Proposed in [RNMFLP: Predicting circRNA–disease associations based on robust nonnegative matrix factorization and label propagation](https://academic.oup.com/bib/article-abstract/23/5/bbac155/6582881?redirectedFrom=fulltext&login=false), Briefings in bioinformatics 2022.

* AE-RF (2021)
    * Proposed in [Inferring Potential CircRNA-Disease Associations via Deep Autoencoder-Based Classification](https://link.springer.com/article/10.1007/s40291-020-00499-y), Molecular diagnosis & therapy 2021.

* DMFCDA (2021)
    * Proposed in [Deep Matrix Factorization Improves Prediction of Human CircRNA-Disease Associations](https://ieeexplore.ieee.org/document/9107417), IEEE Journal of Biomedical and Health Informatics 2021.

* CD-LNLP (2019)
    * Proposed in [Predicting CircRNA-Disease Associations Through Linear Neighborhood Label Propagation Method](https://ieeexplore.ieee.org/document/8731942), IEEE Access 2019.

* RWR (2019)
    * Proposed in [A Model Based on Random Walk with Restart toPredict CircRNA-Disease Associations onHeterogeneous Network](https://ieeexplore.ieee.org/abstract/document/9073607), IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) 2019.

* KATZHCDA (2018)
    * Proposed in [Prediction of CircRNA-Disease Associations Using KATZ Model Based on Heterogeneous Networks](https://www.mdpi.com/2218-273X/12/7/932), Biomolecules 2018.

