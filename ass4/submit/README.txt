###### Assignment 4 NLP Course ######


# Details:
Yehoshua Stern 314963927
Eyal Cohen 207947086


# Requierments:
pickle, sklearn, networkx, spacy.
spacy - recomended to download en_core_web_lg for better score, still the rogram will run with the normal en package


# Important:
our model gets the txt files from the data, not the processed files.


# Train:
python train_model.py train_annotations_file train_txt_file dev_annotations_file dev_txt_file
this program will output the model using pickle to file named "model"


# extract:
python extract.py path_to_txt_file output_file_name
this program will output a file in annotation format with the prediction of the model
runtime is about 5-10 minutes for a text file with 340 sentences, so please be patient.


# eval:
python eval.py gold_annotations_file predictions_annotations_file

