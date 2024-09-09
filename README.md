# Project Repo

This repo contains the files used to conduct the experiments throughout the project.
JTT includes JTT files.
BASS includes BASS files.

The files outside these folders are either from the original Meta repo for LLaMA 3 - this project started with that and this is how we obtained base model results - or for the earlier fine-tuning and preliminary experiments.  A lot of the files are redundant but demostrate some of the testing not shown in the report or presentation such as few-shot learning, prompt configurations, chain-of-thought, PAFT and Mixed-PAFT etc. Since the code is not part of the marking, I have kept these and not tidied up too much - this is code is mainly evidence and to show implementation.

The shell run_script files show an example of how I ran the code. If running the code, the output folders must be created beforehand, seeds set, and you must put in your own HuggingFace details. The scripts are run using torchrun and require the necessary arguments shown in the scripts. The code is written for the Imperial GPU clusters, with a couple of files also adapted for the HPC clusters.

If running the base model without fine-tuning, the scripts use code inspired by the original LLaMA repo so instructions need to be followed from there at https://github.com/meta-llama/llama3.
