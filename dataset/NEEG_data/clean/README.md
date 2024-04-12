# NEEG Script Reasoning Dataset

0) This dataset is released at https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018, here we just reorganize the train/dev/test (140331/10000/10000) datasets and add some remarks on the format.
1) To load the data, use pickle format to load them as binary files. 
2) Each item inside the train/dev/test set comprise of the context event chain (a chain of 8 events), the candidate events (5 in total), and the correct answer index (starting from 0, max 4). 
3) For each event, there are in total 6 entries, namely

        verb, lemmatized_verb, deprel, subj, obj, iobj

    

4) For each event in the same event chain, their main entity is the same, which could be infer from the `deprel` and the `verb`, e.g. if `deprel` is `subj` then the subject is the common entity.