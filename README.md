# Easy Things First

This repository contains the code required to reproduce the results reported in

* Sina Zarrieß, David Schlangen. 2016. Easy Things First: Installments Improve Referring Expression Generation for Objects in Photographs. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)*, Berlin, Germany. 

Doing so requires taking several steps:

1. First, you need to collect the data, and place it where the scripts expect it. This is described below.
2. Then you need to generate refexp for the test set that you want to evaluate on, see [`experiment_run.ipynb`]
   or
   you might want to skip this step, and directly reuse the expression generated for our ACL paper, you can find these
   in the files [`EvalExp1/exp1_ground_truth_generated.csv`, `EvalExp2/exp2_ground_truth_generated.csv`]
3. Now it's time to run the human evaluation, you will need subjects that come to your lab (evaluation GUI is running off-line, at the moment).
4. Run the evaluations in [`results.ipynb`].

While some care has been taken not to hard code too many things, it is not unlikely that you will have to changes some paths.

Questions? Queries? Comments. Email us! (*firstname.lastname*@uni-bielefeld.de)


# Getting the data

### "Segmented and Annotated IAPR TC-12 dataset"

This dataset contains 20k images, mostly holiday-type snaps (outdoor scenes). 

The images themselves are from the "IAPR TC-12" dataset ( <http://www.imageclef.org/photodata>), described in

Grubinger, Clough, Müller, & Deselaers. (2006). The IAPR TC-12 benchmark-a new evaluation resource for visual information systems. International Conference on Language Resources and Evaluation., 13–23.

From this paper: "The majority of the images are provided by viventura, an independent travel company that organizes adventure and language trips to South-America. At least one travel guide accompanies each tour and they maintain a daily online diary to record the adventures and places visited by the tourists (including at least one corresponding photo). Furthermore, the guides provide general photographs of each location, accommodation facilities and ongoing social projects."

We use an augmented distribution of this imageset that also contains region segmentations (hence "Segmented and Annotated IAPR TC-12 dataset").

The dataset is described here: <http://imageclef.org/SIAPRdata>.

But the link to the actual data on that page was dead when I tried it in June 2015; I got the data directly from the first author of the following publication (which gives a further description of the dataset).

[1] Hugo Jair Escalante, Carlos A. Hernández, Jesus A. Gonzalez, A. López-López, Manuel Montes, Eduardo F. Morales, L. Enrique Sucar, Luis Villaseñor and Michael Grubinger.  The Segmented and Annotated IAPR TC-12 Benchmark. Computer Vision and Image Understanding, doi: <http://dx.doi.org/10.1016/j.cviu.2009.03.008>, in press, 2009. 

The directory `saiapr_tc-12` from this distribution needs to be accessible at `Images/SAIAPR/saiapr_tc-12` from the directory of this document.

We also need the `features.mat` from `matlab/`, linked to `saiapr_features.mat`.

### ReferIt/SAIAPR

Using the [ReferIt game](http://tamaraberg.com/referitgame/), Tamara Berg and colleagues collected expressions referring to the regions from the SAIAPR data. The work is described here:

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, Tamara L. Berg.   ReferItGame: Referring to Objects in Photographs of Natural Scenes. Empirical Methods in Natural Language Processing (EMNLP) 2014.  Doha, Qatar.  October 2014. 

The resulting (120k) referring expressions are available from:
<http://tamaraberg.com/referitgame/ReferitData.zip>

They need to be placed at `./ReferItData`.

### Preprocessed Data

Can be found [here](https://www.dropbox.com/sh/h20mxynv3g6zsn3/AABNjMC3OZ29Sk6ZgKznQJuEa?dl=0).

### Directories

This is how your folder should look like:

```
├── EvalExp1
│   ├── evalgui_exp1.py
│   ├── exp1_ground_truth_generated.csv
│   ├── final_logs
│   │   ├── log_8_1508156375.88.txt
│   │   └── log_8_1508156375.88_df.csv
│   └── original_logs_acl2016
├── EvalExp2
│   ├── evalgui_ia.py
│   ├── evalgui_ia_bl.py
│   ├── exp2_ground_truth_generated.csv
│   ├── final_logs
│   └── original_logs_acl2016
├── InData
│   ├── Features 
│   ├── Models 
│   ├── Splits
│   └── wlist.txt 
├── README.md
├── ReferitData 
│   ├── CannedGames.txt
│   ├── README.txt
│   └── RealGames.txt
├── SAIA_Data
│   ├── Models
│   ├── README.txt
│   ├── benchmark
│   ├── escalante09b_preprint.pdf
│   ├── histogram_top100.eps
│   ├── matlab
│   ├── test_set_ground_truth
├── experiment_run.ipynb
├── learningcurves.pdf
├── results.ipynb
├── wacgen.py
├── wacgen_ia.py
``

# Running the Experiment

* look at **experiment_run.ipynb**

this contains the instructions on how to prepare and compile the data
for running the experiments


# Running the Evaluation

* look at **results.ipynb**
