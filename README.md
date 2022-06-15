This is the evaluation code on the entailment datasets for the following papers:

Learning Typed Entailment Graphs with Global Soft Constraints, Mohammad Javad Hosseini, Nathanael Chambers, Siva Reddy, Xavier Holt, Shay B. Cohen, Mark Johnson, and Mark Steedman. Transactions of the Association for Computational Linguistics (TACL 2018).

Duality of Link Prediction and Entailment Graph Induction,  Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Association for Computational Linguistics (ACL 2019).

Open-Domain Contextual Link Prediction and its Complementarity with Entailment Graphs, Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Findings of the Association for Computational Linguistics: EMNLP 2021.

## Levy/Holt's Dataset

The Levy/Holt's dataset along with its splits can be found in the LevyHoltDS folder. The folder contains dev (the development set), test (the test set), dev_dir (the directional portion of the dev set), and test_dir (the directional portion of the test set). Please note:

- The format is: "triple_q[TAB]triple_p[TAB]label", where label (either True or False) means whether triple_p entails triple_q (TRUE) or not (FALSE). **Please note that the order of the triples is reverse, similar to the Levy and Dagan (2016) dataset, i.e., p=>q, where p is the right-hand side predicate and q is the left-hand-side predicate.**

- Each partition has 3 files, e.g., the dev partition contains:
  - dev_s.txt: The human readable format: "arg1,q,arg2[TAB]arg1,p,arg2[TAB]label". The order is p=>q as mentioned above. The argument order of the second triple is sometimes the reverse order of the first triple.
  - dev.txt. The original file, where each line has: "type1, q, arg2[TAB]arg1, p, arg2[TAB]label. This is again based on Levy and Dagan (2016) dataset.
  - dev_rels.txt: The parsed relations based on a CCG parser: "CCG_q arg1:type1 arg2:type2[TAB]CCG_p arg1:type1 arg2:type2[TAB]label". The order is p=>q as mentioned above. The argument order of the second triple is sometimes the reverse order of the first triple. The types are based on FIGER first level types.

- The directional portions contain example entailment pairs where (p ENTAILS q) and (q DOES NOT ENTAIL p). A symmetric model will have the precision exactly .5 on these portions.

- The experiments for this paper are only run on dev and test. Please see our 2021 paper for experiments on the directional portions: [[paper]](https://aclanthology.org/2021.findings-emnlp.238.pdf)
  
- The code below will download the dataset separately. So you can run the code without using the LevyHoltDS directory.

## How to Run the Code

Please follow the below instructions to create entailment graphs and/or replicate the paper's experiments.

### Step 1: Clone the entgraph_eval project and download necessary files.

    git clone https://github.com/mjhosseini/entgraph_eval.git
    wget https://dl.dropboxusercontent.com/s/j7sgqhp8a27qgcf/gfiles.zip
    unzip gfiles.zip
    rm gfiles.zip

***The datasets can be found in the following files inside the folder 'gfiles/ent':***

A) Levy/Holt's dev set: dev.txt and dev_rels.txt

B) Levy/Holt's test set: test.txt and test_rels.txt

C) Berant's dataset: ber_all.txt and ber_all_rels.txt

Please note:

1) The *_rels.txt files contain the extracted CCG triples while the other ones are the original triples.

2) In all these files, the format is: "hypothesis_triple[TAB]premise_triple[TAB]label". Note that the first triple is the hypothesis triple (not the premise) which is based on the format in Levy and Dagan (2016) dataset paper (Inference in Context via Question Answering, ACL 2016).

3) We performed the split of the Levy/Holt's dataset as explained in Section 5.2 of the TACL 2018 paper. We use the dev set for hyper-parameter tuning.

4) The rest of the files in gfiles/ent (e.g., all_*) are for faster evaluation (relevant to Step 5 below).

### Step 2: Add the learned entailment graphs folder inside the gfiles folder.
You can download and unzip learned global_graphs from https://worksheets.codalab.org/worksheets/0x8684ad8e95e24c4d80074278bce37ba4.

### Step 3: Install dependencies (if required)

    pip install numpy
    pip install scipy
    pip install nltk
    python -m nltk.downloader wordnet
    python -m nltk.downloader verbnet
    python -m nltk.downloader stopwords
    pip install sklearn

### Step 4: Run the evaluation script.

    cd entgraph_eval/evaluation/
    python eval.py --gpath global_graphs --dev --sim_suffix _gsim.txt --method global_scores --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 1 --exactType --backupAvg --write
    python eval.py --gpath global_graphs --dev --sim_suffix _gsim.txt --method local_scores --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --exactType --write
    
The code outputs a file called gfiles/results/pr_rec/method_name.txt (e.g., gfiles/results/pr_rec/global_scores.txt or gfiles/results/pr_rec/local_scores.txt). This file contains the precisions and recalls that are obtained by changing a threshold on the entailment scores. It also contains the area under precision-recall curve (for precision>=.5). The code also writes the precisions and recalls of a few baselines as well as the precision and recall for the entailment graphs at precision~0.75.

The main parameters are these ones:

--gpath: The folder containing entailment graphs, which should be put inside gfiles folder.

--sim_suffix: The suffix of the entailment graph files, e.g., _gsim.txt.

-- featIdx 1: There are usually more than one similarity measures in the entailment graph files (e.g., local similarity and global similarity in the global_scores folder). This index specifies which similarity measure should be used. For "global_scores", 0 means local and 1 means global.

--exactType --backupAvg: If you add these two options together, the code first tries to use the similarity measure of the graph with the same types as the entailment query. For example, for (PERSON visit LOCATION) => (PERSON arrive in LOCATION), it will use the similarity measure for the (PERSON,LOCATION) graph. If that graph doesn't have the relations of interest (visit or arrive in), then the code looks at the uniform average of the scores for those relations across all graphs (the main results with global similarities in the papers). If only --exactType is used, then the similarity will be 0 if the graph doesn't have the relations of interest (the local results in the papers). Finally, if none of these options are used, the code always uses the uniform avarage of the similarity scores across all graphs (the avg results in the first paper).

--method global_scores: A given name to the similarity measures (e.g., global_scores in our case).

Other parameters that should mainly remain unchanged:

--CCG 1: Evaluate based on entailment graphs and datasets with CCG parser extractions (0 means openIE extractions). Our preleminary experiments showed better results with the CCG extractions.

--typed 1: Use the types of arguments. If set to 0, it will ignore all the types, but the entailment graph should also be untyped (e.g., only thing#thing_sim.txt as the only entailment graph).

--supervied 0: All the experiments are unsupervised.

--useSims 0: If we set this to 1, the code will look into other relations that have high similarity measures to the relations of interest in GloVe embedding space. We did not use this option in the above papers, but using it will improve the results slightly. If you're interested in testing other embedding spaces, you can do that by providing a file in the format of gfiles/ent/ccg.sim.

--oneFeat 1: This means that we only use one of the similarity measures and don't combine them in any way.

--no_lemma_baseline: It won't run the lemma_baseline in advance. All the results in the papers are WITH the lemma baseline.

--no_constraints: It won't use the extra constraints in advance. All the results in the papers are WITH the additional constraints.

--debug: More information will be written out.

For more options, please take a look at constants/flags.py.

### Step 5: (Optional)

The above code will take a few hours to run. The reason is that for the provided entailment datasets, it will look into all the entailment graphs to extract the relevant similarity features (e.g., see feats/feats_global_scores.txt). If you run the code once on some entailment graphs and are interested in changing some of the options (e.g., the featIdx, or --test instead of --dev), you don't need to run the whole code again. You can simply run the script below, which only uses the extracted similarity measures for those entailment datasets. This code just takes a few minutes to complete. In that case, it's suggested to run step 4 with --useSims 1, because otherwise this option can't be turned on in this step.

First copy the extracted similarity features.

    cp feats/feats_global_scores.txt ../../gfiles/ent/

Then, run the below code.

    python eval.py --featsFile feats_global_scores --dev --method global_scores --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 1 --exactType --backupAvg --write

There is only one new parameter:

--featsFile feats_global_scores: The name of the file that contains extracted similarity features of the datasets.
