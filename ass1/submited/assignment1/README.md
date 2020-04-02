How to run code for task memm-1:
the two programs, ExtractFeatures.py and TrainSolver.py, run as follows:

python3 ExtractFeatures.py corpus_file features_file feature_mapping_file
python3 TrainSolver.py features_file model_file feature_mapping_file

where corpus_file, features_file and model_file are as described in the assignment, 
and feature_mapping_file is a file which we use to save the feature mapping for the 
model trained in TrainSolver, and is additionally used for saving the sets of possible tags for each word (or word signature), which will be used to speed up the implementation of the viterbi memm.
if no argument for feature_mapping_file is provided, the programs will default to the file "feature_mapping".

