import os
import re
import json

import nltk
import string
#nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def tag_token(tags, token_pos, tag):
	if token_pos > 0 and f'{tag}' in tags[token_pos - 1]:
		
		if tags[token_pos] == 'O':
			tags[token_pos] = f'I-{tag}'
		elif f'I-{tag}' not in tags[token_pos]:
			tags[token_pos] += f';I-{tag}'
	else:
		if tags[token_pos] == 'O':
			tags[token_pos] = f'B-{tag}'
		elif f'B-{tag}' not in tags[token_pos]:
			tags[token_pos] += f';B-{tag}'


def remove_trailing_punctuation(token):
	while token and (re.search(r'[^\w\s\']', token[-1]) or (len(token) > 1 and token[-1] == "'")):
		token = token[:-1]
		
	return token


class AnnToBioConverter:
	
	def __init__(self, input_dir=None, txt_dir=None, ann_dir=None, output_dir=None, filtered_entities=[]):
		"""
		Initializes an instance of the AnnToBioConverter class.
		:param input_dir: (str) Directory containing both the text and annotation files.
		:param txt_dir: (str) Directory containing the text files.
		:param ann_dir: (str) Directory containing the annotation files.
		:param output_dir: (str) Directory where the JSON file will be saved.
		"""
		if input_dir is not None:
			self.txt_dir = input_dir
			self.ann_dir = input_dir
		elif txt_dir is not None and ann_dir is not None:
			self.txt_dir = txt_dir
			self.ann_dir = ann_dir
		else:
			raise ValueError("Either input_dir or txt_dir and ann_dir must be provided.")
		
		if output_dir is not None:
			self.output_dir = output_dir
		else:
			raise ValueError("output_dir must be provided.")
		self.data = {}
		
		self.filtered_entities = filtered_entities
	
	def _splitting_tokens(self, file_id, start, end, hyphen_split):
		"""
		Splits a multi-word token into separate tokens and returns a list of tokens and their respective start and end indices.
		:param file_id: (str) The ID of the file containing the token.
		:param start: (int) The starting index of the token in the text.
		:param end: (int) The ending index of the token in the text.
		:return: (tuple) A tuple containing a list of tokens and their respective start and end indices.
		"""
		
		text = self.data[file_id]['text']
		token = text[start:end]
		
		extra_sep = ['\u200a']
		if hyphen_split:
			extra_sep += ['-', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212', '\uff0d']
		
		new_range = []
		tokens = []
		
		curr = start
		new_start = None
		
		for c in token + " ":
			if c.isspace() or c in extra_sep:
				if new_start:
					new_range.append([new_start, curr])
					tokens.append(text[new_start:curr])
					new_start = None
			elif not new_start:
				new_start = curr
			curr += 1
		
		return tokens, new_range
	
	def _load_txt(self):
		"""
		Loads the text files into the instance's data dictionary.
		"""
		for file_name in os.listdir(self.txt_dir):
			if file_name.endswith(".txt"):
				with open(os.path.join(self.txt_dir, file_name), "r") as f:
					text = f.read()
				file_id = file_name.split('.')[0]
				self.data[file_id] = {
					"text": text,
					"annotations": []
				}
	
	def _load_ann(self, hyphen_split):
		for file_name in os.listdir(self.ann_dir):
			
			if file_name.endswith(".ann"):
				with open(os.path.join(self.ann_dir, file_name), "r") as f:
					
					file_id = file_name.split('.')[0]
					annotations = []
					
					for line in f:
						if line.startswith("T"):
							fields = line.strip().split("\t")
							if len(fields[1].split(" ")) > 1:
								label = fields[1].split(" ")[0]
								
								# Extracting start end indices (Few annotations contain more than one disjoint ranges)
								start_end_range = [
									list(map(int, start_end.split()))
									for start_end in ' '.join(fields[1].split(" ")[1:]).split(';')
								]
								
								start_end_range_fixed = []
								for start, end in start_end_range:
									tokens, start_end_split_list = self._splitting_tokens(file_id, start, end,
																						  hyphen_split)
									start_end_range_fixed.extend(start_end_split_list)
								
								# Adding labels, start, end to annotations
								for start, end in start_end_range_fixed:
									annotations.append({
										"label": label,
										"start": start,
										"end": end
									})
					# sort annotations based on 'start' key before adding it to our dataset
					annotations = sorted(annotations, key=lambda x: (x['start'], x['label']))
					self.data[file_id]["annotations"] = annotations
		self._manual_fix()
	
	def split_text(self, file_id):
		text = self.data[file_id]['text']
		regex_match = r'[^\s\u200a\-\u2010-\u2015\u2212\uff0d]+'  # r'[^\s\u200a\-\—\–]+'
		
		tokens = []
		start_end_ranges = []
		
		sentence_breaks = []
		
		start_idx = 0
		
		for sentence in text.split('\n'):
			words = [match.group(0) for match in re.finditer(regex_match, sentence)]
			processed_words = list(map(remove_trailing_punctuation, words))
			sentence_indices = [(match.start(), match.start() + len(token)) for match, token in
								zip(re.finditer(regex_match, sentence), processed_words)]
			
			# Update the indices to account for the current sentence's position in the entire text
			sentence_indices = [(start_idx + start, start_idx + end) for start, end in sentence_indices]
			
			start_end_ranges.extend(sentence_indices)
			tokens.extend(processed_words)
			
			sentence_breaks.append(len(tokens))
			
			start_idx += len(sentence) + 1
		return tokens, start_end_ranges, sentence_breaks
	
	def write_bio_files(self):
		
		for file_id in self.data:
			text = self.data[file_id]['text']
			annotations = self.data[file_id]['annotations']
			
			# Tokenizing
			tokens, token2text, sentence_breaks = self.split_text(file_id)
			
			# Initialize the tags
			tags = ['O'] * len(tokens)
			
			ann_pos = 0
			token_pos = 0
			
			while ann_pos < len(annotations) and token_pos < len(tokens):
				
				label = annotations[ann_pos]['label']
				start = annotations[ann_pos]['start']
				end = annotations[ann_pos]['end']
				
				if self.filtered_entities:
					if label not in self.filtered_entities:
						# increment to access next annotation
						ann_pos += 1
						continue
				
				ann_word = text[start:end]
				
				# find the next word that fall between the annotation start and end
				while token_pos < len(tokens) and token2text[token_pos][0] < start:
					token_pos += 1
				
				if tokens[token_pos] == ann_word or \
					ann_word in tokens[token_pos] or \
					re.sub(r'\W+', '', ann_word) in re.sub(r'\W+', '', tokens[token_pos]):
					tag_token(tags, token_pos, label)
				elif ann_word in tokens[token_pos - 1] or \
					ann_word in tokens[token_pos - 1] or \
					re.sub(r'\W+', '', ann_word) in re.sub(r'\W+', '', tokens[token_pos - 1]):
					tag_token(tags, token_pos - 1, label)
				else:
					print(tokens[token_pos], tokens[token_pos - 1], ann_word, label)
				
				# increment to access next annotation
				ann_pos += 1
			
			# Write the tags to a .bio file
			with open(os.path.join(self.output_dir, f"{file_id}.bio"), 'w') as f:
				for i in range(len(tokens)):
					token = tokens[i].strip()
					if token:
						if i in sentence_breaks:
							f.write("\n")
						f.write(f"{tokens[i]}\t{tags[i]}\n")
	
	def _manual_fix(self):
		fix = {
			'19214295': {
				425: 424
			}
		}
		for file_id in self.data:
			if file_id in fix:
				for ann in self.data[file_id]['annotations']:
					if ann['start'] in fix[file_id]:
						ann['start'] = fix[file_id][ann['start']]
	
	def load_data(self, reload=False, hyphen_split=False):
		if not self.data or reload:
			self.data = {}
			self._load_txt()
			self._load_ann(hyphen_split)
	
	def create_json(self, data_dir, reload=False, hyphen_split=False):
		if not self.data or reload:
			self.load_data(hyphen_split, reload)
		
		# Write the dictionary to a JSON file
		with open(os.path.join(data_dir, "data.json"), "w") as f:
			json.dump(self.data, f)
	
	def load(self):
		self.load_txt()
		self.load_ann()
	
	def __getattr__(self, name):
		if name == 'data':
			return self.data
		else:
			raise AttributeError(f"'AnnToBioConverter' object has no attribute '{name}'")
