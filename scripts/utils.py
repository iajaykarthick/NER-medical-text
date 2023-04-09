import re
import os
import pickle

import spacy
from spacy import displacy

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')

# Load the tokenizer from file
with open('../data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def load_data(data_dir):
	data = np.load(os.path.join(data_dir, 'data.npz'), allow_pickle=True)
	
	train_sequences_padded = data['train_sequences_padded']
	train_labels = data['train_labels']
	
	val_sequences_padded = data['val_sequences_padded']
	val_labels = data['val_labels']
	
	test_sequences_padded = data['test_sequences_padded']
	test_labels = data['test_labels']
	
	label_to_index = data['label_to_index'].item()  # use .item() to convert the numpy array to a Python dictionary
	
	index_to_label = data['index_to_label'].item()
	
	return (train_sequences_padded, train_labels), (val_sequences_padded, val_labels), (
	test_sequences_padded, test_labels), label_to_index, index_to_label


def clean_word(word):
	"""
	Cleans a word by removing non-alphanumeric characters and extra whitespaces,
	converting it to lowercase, and checking if it is a stopword.

	Args:
	- word (str): the word to clean

	Returns:
	- str: the cleaned word, or an empty string if it is a stopword
	"""
	# remove non-alphanumeric characters and extra whitespaces
	word = re.sub(r'[^\w\s]', '', word)
	word = re.sub(r'\s+', ' ', word)
	
	# convert to lowercase
	word = word.lower()
	
	if word not in STOP_WORDS:
		return word
	
	return ''

def tokenize_text(text):
	"""
    Tokenizes a text into a list of cleaned words.

    Args:
    - text (str): the text to tokenize

    Returns:
    - tokens (list of str): the list of cleaned words
    - start_end_ranges (list of tuples): the start and end character positions for each token
    """
	regex_match = r'[^\s\u200a\-\u2010-\u2015\u2212\uff0d]+'  # r'[^\s\u200a\-\—\–]+'
	tokens = []
	start_end_ranges = []
	# Tokenize the sentences in the text
	sentences = nltk.sent_tokenize(text)
	
	start = 0
	for sentence in sentences:
		
		sentence_tokens = re.findall(regex_match, sentence)
		curr_sent_tokens = []
		curr_sent_ranges = []
		
		for word in sentence_tokens:
			word = clean_word(word)
			if word.strip():
				start = text.lower().find(word, start)
				end = start + len(word)
				curr_sent_ranges.append((start, end))
				curr_sent_tokens.append(word)
				start = end
		if len(curr_sent_tokens) > 0:
			tokens.append(curr_sent_tokens)
			start_end_ranges.append(curr_sent_ranges)
			
	return tokens, start_end_ranges

# def tokenize_text(text):
# 	"""
# 	Tokenizes a text into a list of cleaned words.
#
# 	Args:
# 	- text (str): the text to tokenize
#
# 	Returns:
# 	- list of str: the list of cleaned words
# 	"""
# 	regex_match = r'[^\s\u200a\-\u2010-\u2015\u2212\uff0d]+'  # r'[^\s\u200a\-\—\–]+'
# 	tokens = []
# 	for sentence in text.split('\n'):
# 		sentence_tokens = re.findall(regex_match, sentence)
# 		for word in sentence_tokens:
# 			word = clean_word(word)
# 			if word.strip():
# 				tokens.append(word)
# 	return tokens


def predict(text, model, index_to_label, acronyms_to_entities, MAX_LENGTH):
	"""
	Predicts named entities in a text using a trained NER model.

	Args:
	- text (str): the text to predict named entities in
	- model: the trained NER model
	- tokenizer: the trained tokenizer used for the model
	- index_to_label (list of str): a list mapping each index in the predicted sequence to a named entity label
	- acronyms_to_entities (dict): a dictionary mapping acronyms to their corresponding named entity labels
	- MAX_LENGTH (int): the maximum sequence length for the model

	Returns:
	- None
	"""
	
	tokens, start_end_ranges = tokenize_text(text)
	all_tokens = []
	all_ranges = []
	for sent_tokens, sent_ranges in zip(tokens, start_end_ranges):
		for token, start_end in zip(sent_tokens, sent_ranges):
			start, end = start_end[0], start_end[1]
			all_tokens.append(token)
			all_ranges.append((start, end))
			
	sequence = tokenizer.texts_to_sequences([' '.join(token for token in all_tokens)])
	padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
	
	# Make the prediction
	prediction = model.predict(np.array(padded_sequence))
	
	# Decode the prediction
	predicted_labels = np.argmax(prediction, axis=-1)
	predicted_labels = [index_to_label[i] for i in predicted_labels[0]]
	
	entities = []
	start_char = 0
	for i, (token, label, start_end_range) in enumerate(zip(all_tokens, predicted_labels, all_ranges)):
	
		start = start_end_range[0]
		end = start_end_range[1]
		
		if label != 'O':
			entity_type = acronyms_to_entities[label[2:]]
			entity = (start, end, entity_type)
			entities.append(entity)
	
	# Print the predicted named entities
	print("Predicted Named Entities:")
	for i in range(len(all_tokens)):
		if predicted_labels[i] == 'O':
			print(f"{all_tokens[i]}: {predicted_labels[i]}")
		else:
			print(f"{all_tokens[i]}: {acronyms_to_entities[predicted_labels[i][2:]]}")
	
	display_pred(text, entities)

def display_pred(text, entities):
	nlp = spacy.load("en_core_web_sm", disable=['ner'])
	# Generate the entities in Spacy format
	doc = nlp(text)
	# Add the predicted named entities to the Doc object
	for start, end, label in entities:
		span = doc.char_span(start, end, label=label)
		if span is not None:
			doc.ents += tuple([span])
	
	colors = {"Activity": "#f9d5e5",
			  "Administration": "#f7a399",
			  "Age": "#f6c3d0",
			  "Area": "#fde2e4",
			  "Biological_attribute": "#d5f5e3",
			  "Biological_structure": "#9ddfd3",
			  "Clinical_event": "#77c5d5",
			  "Color": "#a0ced9",
			  "Coreference": "#e3b5a4",
			  "Date": "#f1f0d2",
			  "Detailed_description": "#ffb347",
			  "Diagnostic_procedure": "#c5b4e3",
			  "Disease_disorder": "#c4b7ea",
			  "Distance": "#bde0fe",
			  "Dosage": "#b9e8d8",
			  "Duration": "#ffdfba",
			  "Family_history": "#e6ccb2",
			  "Frequency": "#e9d8a6",
			  "Height": "#f2eecb",
			  "History": "#e2f0cb",
			  "Lab_value": "#f4b3c2",
			  "Mass": "#f4c4c3",
			  "Medication": "#f9d5e5",
			  "Nonbiological_location": "#f7a399",
			  "Occupation": "#f6c3d0",
			  "Other_entity": "#d5f5e3",
			  "Other_event": "#9ddfd3",
			  "Outcome": "#77c5d5",
			  "Personal_background": "#a0ced9",
			  "Qualitative_concept": "#e3b5a4",
			  "Quantitative_concept": "#f1f0d2",
			  "Severity": "#ffb347",
			  "Sex": "#c5b4e3",
			  "Shape": "#c4b7ea",
			  "Sign_symptom": "#bde0fe",
			  "Subject": "#b9e8d8",
			  "Texture": "#ffdfba",
			  "Therapeutic_procedure": "#e6ccb2",
			  "Time": "#e9d8a6",
			  "Volume": "#f2eecb",
			  "Weight": "#e2f0cb"}
	options = {"compact": True, "bg": "#F8F8F8",
			   "ents": list(colors.keys()),
			   "colors": colors}
	
	# Generate the HTML visualization
	html = displacy.render(doc, style="ent", options=options)

# def predict(text, model, tokenizer, index_to_label, acronyms_to_entities, MAX_LENGTH):
# 	"""
# 	Predicts named entities in a text using a trained NER model.
#
# 	Args:
# 	- text (str): the text to predict named entities in
# 	- model: the trained NER model
# 	- tokenizer: the trained tokenizer used for the model
# 	- index_to_label (list of str): a list mapping each index in the predicted sequence to a named entity label
# 	- acronyms_to_entities (dict): a dictionary mapping acronyms to their corresponding named entity labels
# 	- MAX_LENGTH (int): the maximum sequence length for the model
#
# 	Returns:
# 	- None
# 	"""
#
# 	tokens = tokenize_text(text)
# 	sequence = tokenizer.texts_to_sequences([' '.join(token for token in tokens)])
# 	padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
#
# 	# Make the prediction
# 	prediction = model.predict(np.array(padded_sequence))
#
# 	# Decode the prediction
# 	predicted_labels = np.argmax(prediction, axis=-1)
# 	predicted_labels = [index_to_label[i] for i in predicted_labels[0]]
#
# 	# Print the predicted named entities
# 	print("Predicted Named Entities:")
# 	for i in range(len(tokens)):
# 		if predicted_labels[i] == 'O':
# 			print(f"{tokens[i]}: {predicted_labels[i]}")
# 		else:
# 			print(f"{tokens[i]}: {acronyms_to_entities[predicted_labels[i][2:]]}")
#

def predict_multi_line_text(text, model, index_to_label, acronyms_to_entities, MAX_LENGTH):
	
	# sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
	# sent_tokens = []
	# sent_start_end = []
	sequences = []
	
	sent_tokens, sent_start_end = tokenize_text(text)
	
	for i in range(len(sent_tokens)):
		sequence = tokenizer.texts_to_sequences([' '.join(token for token in sent_tokens[i])])
		sequences.extend(sequence)
	
	# for sentence in sentences:
	# 	tokens, start_end_ranges = tokenize_text(sentence)
	# 	sequence = tokenizer.texts_to_sequences([' '.join(token for token in tokens)])
	# 	sequences.append(sequence[0])
	# 	sent_tokens.append(tokens)
	# 	sent_start_end.append(start_end_ranges)
		
	padded_sequence = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
	
	# Make the prediction
	prediction = model.predict(np.array(padded_sequence))
	
	# Decode the prediction
	predicted_labels = np.argmax(prediction, axis=-1)
	
	predicted_labels = [
		[index_to_label[i] for i in sent_predicted_labels]
		for sent_predicted_labels in predicted_labels
	]
	
	entities = []
	start_char = 0
	
	for tokens, sent_pred_labels, start_end_ranges in zip(sent_tokens, predicted_labels, sent_start_end):
		
		for i, (token, label, start_end_range) in enumerate(zip(tokens, sent_pred_labels, start_end_ranges)):
			start = start_end_range[0]
			end = start_end_range[1]
			
			if label != 'O':
				entity_type = acronyms_to_entities[label[2:]]
				entity = (start, end, entity_type)
				entities.append(entity)
		
	# Print the predicted named entities
	print("Predicted Named Entities:")
	for i in range(len(sent_tokens)):
		for j in range(len(sent_tokens[i])):
			if predicted_labels[i][j] == 'O':
				print(f"{sent_tokens[i][j]}: {predicted_labels[i][j]}")
			else:
				print(f"{sent_tokens[i][j]}: {acronyms_to_entities[predicted_labels[i][j][2:]]}")
		print("\n\n\n")
	
	display_pred(text, entities)
	# return entities