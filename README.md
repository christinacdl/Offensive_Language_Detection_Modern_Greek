# Offensive_Language_Detection_Modern_Greek

We introduce a new corpus, named AIKIA, for Offensive Language Detection (OLD) in Modern Greek (EL). EL is a less-resourced language regarding OLD. AIKIA offers free access to annotated data leveraged from EL Twitter and fiction texts using the lexicon of offensive terms, ERIS, that originates from HurtLex. AIKIA has been annotated for offensive values with the Best Worst Scaling (BWS) method, which is designed to avoid problems of categorical and scalar annotation methods. BWS assigns continuous offensive scores in the form of floating point numbers instead of binary arithmetical or categorical values. AIKIA’s performance in OLD was tested by fine-tuning a variety of pre-trained language models in a binary classification task. Experimentation with a number of thresholds showed that the best mapping of the continuous values to binary labels should occur at the range [0.5-0.6] of BWS values and that the pre-trained models on EL data achieved the highest Macro-F1 scores. Greek-Media-BERT outperformed all models with a threshold of 0.6 by obtaining a Macro-F1 score of 0.92

AIKIA Corpus available here: https://osf.io/vae2u/?view_only=d21e6fdc5ffc4ac794d4b2c5972d2742 

Paper: https://aclanthology.org/2024.lrec-main.1378
