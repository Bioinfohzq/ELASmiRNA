# ELASmiRNA
ELASmiRNA is a stacked ensemble learning model for accurate prediction of abiotic stress-responsive miRNAs.
# Overview
Features Kmer and DAC are extracted using pse-in-one 2.0. The code "autogluon.py" is used for model training and performance evaluation. For the output results, we selected 8 indicators for evaluation, namely: SE, SPC, ACC, F1-score, PPV, NPV, MCC, and AUC.File Feature is the extracted Kmer and DAC features of miRNA and PremiRNA
# Dependency
python3.9

windows/linux

AutoGluon

sklearn
# Datasets
The file "Datasets" is the RNA sequence dataset used in this study.A total of four datasets were miRNA, Pre-miRNA, miRNA+Pre-miRNA and Independent_dataset.miRNA contained 376 stress-responsive mirnas and 376 stress-non-responsive mirnas. Pre-mirnas included 251 stress-responsive pre-mirnas and 251 stress-non-responsive pre-mirnas. miRNA+Pre-miRNA contains 251 stress-responsive mirnas and 251 stress-responsive pre-mirnas as well as 251 stress-unresponsive mirnas and 251 stress-unresponsive pre-mirnas. Independent_dataset included 74 stress-responsive mirnas and 100 stress-non-responsive mirnas. There were 74 stress-responsive pre-mirnas and 100 stress-non-responsive pre-mirnas and miRNA+ pre-mirnas (74 stress-responsive mirnas and 100 stress-non-responsive mirnas; 74 stress-responsive pre-mirnas and 100 stress-non-responsive pre-mirnas). The stress-responsive miRNA, Pre-miRNA and miRNA+Pre-miRNA were positive classes. Stress non-responsive miRNA, Pre-miRNA, and miRNA+Pre-miRNA were in the negative category.
# Usage
First, you should download pse-in-one 2.0(http://bioinformatics.hitsz.edu.cn/Pse-in-One2.0/server/) to extract features Kmer (K =2, 3, 4, 5), DAC (lag =3).Finally,The code "autogluon.py" is used for model training and performance evaluation. 
