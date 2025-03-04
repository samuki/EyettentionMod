#coding=utf-8
import re
import os
import numpy as np
import pickle
import pandas as pd
from typing import Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

from model import Eyettention

def save_with_pickle(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)
        
def load_with_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
        
def load_pretrained_model(model_path, cf, device):
	# Load model
	dnn = Eyettention(cf)
	dnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)),  strict=False)
	return dnn

def load_bsc() -> Tuple[pd.DataFrame, ...]:
	"""
	:return: word info dataframe, part-of-speech info, eye movements
	"""
	bsc_path = './Data/beijing-sentence-corpus/'
	info_path = os.path.join(bsc_path, 'BSC.Word.Info.v2.xlsx')
	bsc_emd_path = os.path.join(bsc_path, 'BSC.EMD/BSC.EMD.txt')
	word_info_df = pd.read_excel(info_path, 'word')
	pos_info_df = pd.read_excel(info_path, header=None)
	eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')
	return word_info_df, pos_info_df, eyemovement_df

def load_corpus(corpus, task=None):
	if corpus == 'BSC':
		#load word data, POS data, EM data
		word_info_df, pos_info_df, eyemovement_df = load_bsc()
		return word_info_df, pos_info_df, eyemovement_df

	elif corpus == 'celer':
		eyemovement_df = pd.read_csv('./Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t')
		eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)', '', regex=True)
		word_info_df = pd.read_csv('./Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
		word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
		return word_info_df, None, eyemovement_df

	elif corpus == 'meco':
		import pyreadr
		# Reading the R dataset
		eye_tracking_path = 'Data/meco/joint_fix_trimmed_l1.rda'  # or 'Data/meco/joint_fix_trimmed_l2.rda'
		texts_path = 'Data/meco/supp_texts_l1.csv'
		meco_rda = pyreadr.read_r(eye_tracking_path)
		# Extracting the DataFrame
		meco_df = meco_rda["joint.fix"]
		# Creating a list of sentences 5 x 14,
		sentences_df = pd.read_csv(texts_path)
		language_code = {
			'Dutch': 'du',
			'English': 'en',
			'Estonian': 'ee',
			'Finnish': 'fi',
			'German': 'ge',
			'Greek': 'gr',
			'Hebrew': 'he',
			'Italian': 'it',
			'Korean': 'ko',
			'Norwegian': 'no',
			'Russian': 'ru',
			'Spanish': 'sp',
			'Turkish': 'tr'
		}
		# Converting the language column to codes
		sentences_df['Unnamed: 0'] = sentences_df['Unnamed: 0'].map(language_code)
		# Renaming the language column
		sentences_df.rename(columns={'Unnamed: 0': 'Language'}, inplace=True)
		# Renaming the text columns to numeric indices
		text_columns = sentences_df.columns[1:-2]  # Excluding the first language column and last two unnamed columns
		renamed_columns = {name: str(index) for index, name in enumerate(text_columns, start=1)}
		sentences_df.rename(columns=renamed_columns, inplace=True)
		# Dropping the last two unnamed columns
		sentences_df.drop(columns=['Unnamed: 13', 'Unnamed: 14'], inplace=True)
		# Final 13 x 13 df
		sentences_df = sentences_df.dropna(how='all')
		# Creating a list of unique readers (subjects)
		readers = list(meco_df['uniform_id'].dropna().unique())
		return meco_df, sentences_df, readers


def compute_BSC_word_length(sn_df):
	word_len = sn_df.LEN.values
	wl_list = []
	for wl in word_len:
		wl_list.extend([wl]*wl)
	arr = np.asarray(wl_list, dtype=np.float32)
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr


def pad_seq(seqs, max_len, pad_value, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=pad_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 0] = 0
		padded[i, 1:(len(seq)+1)] = seq
		if pad_value !=0:
			padded[i, len(seq)+1] = pad_value -1

	return padded

def pad_seq_with_nan(seqs, max_len, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=np.nan, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 1:(len(seq)+1)] = seq
	return padded

def _process_BSC_corpus(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf):
	"""
	SN_token_embedding   <CLS>, bla, bla, <SEP>
	SP_token_embedding   <CLS>, bla, bla, <SEP>
	SP_ordinal_pos 0, bla, bla, max_sp_len
	SP_fix_dur     0, bla, bla, 0
	SN_len         original sentence length without start and end tokens
	"""
	SN_input_ids, SN_attention_mask, SN_WORD_len = [], [], []
	SP_input_ids, SP_attention_mask = [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list = []
	for sn_id in sn_list:
		#print('sub_id:', sub_id)
		#process sentence sequence
		sn_df = eyemovement_df[eyemovement_df.sn==sn_id]
		sn = word_info_df[word_info_df.SN == sn_id]
		sn_str = ''.join(sn.WORD.values)
		sn_word_len = compute_BSC_word_length(sn)

		#tokenization and padding
		tokenizer.padding_side = 'right'
		tokens = tokenizer.encode_plus(sn_str,
										add_special_tokens = True,
										truncation=True,
										max_length = cf["max_sn_len"],
										padding = 'max_length',
										return_attention_mask=True)
		encoded_sn = tokens["input_ids"]
		mask_sn = tokens["attention_mask"]

		#process fixation sequence
		for sub_id in reader_list:
			sub_df = sn_df[sn_df.id==sub_id]
			if len(sub_df) == 0:
				#no scanpath data found for the subject
				continue

			#last fixation go back to the first character with fl = 0 -- seems to be outlier point? remove
			if sub_df.iloc[-1].wn == 1 and sub_df.iloc[-1].fl == 0:
				sub_df = sub_df.iloc[:-1]

			sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.wn.values, sub_df.fl.values, sub_df.dur.values
			sp_landing_pos_char = np.modf(sp_fix_loc)[0]
			SP_landing_pos.append(sp_landing_pos_char)

			#Convert word-based ordinal positions to token(character)-based ordinal positions
			#When the fixated word index is less than 0, set it to 0
			sp_fix_loc = np.where(sp_fix_loc<0, 0, sp_fix_loc)
			sp_ordinal_pos = [np.sum(sn[sn.NW<value].LEN) + np.ceil(sp_fix_loc[count]+ 1e-10) for count, value in enumerate(sp_word_pos)]
			SP_ordinal_pos.append(sp_ordinal_pos)
			SP_fix_dur.append(sp_fix_dur)

			#tokenization and padding for scanpath, i.e. fixated word sequence
			sp_token = [sn_str[int(i-1)] for i in sp_ordinal_pos]
			sp_token_str = '[CLS]' + ''.join(sp_token) + '[SEP]'
			sp_tokens = tokenizer.encode_plus(sp_token_str,
												add_special_tokens = False,
												truncation=True,
												max_length = cf["max_sp_len"],
												padding = 'max_length',
												return_attention_mask=True)
			encoded_sp = sp_tokens["input_ids"]
			mask_sp = sp_tokens["attention_mask"]
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)

			#sentence information
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(sn_word_len)
			sub_id_list.append(sub_id)

	#padding for batch computation
	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list, dtype=np.int64)

	data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len,
			"SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask,
			"SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
			"sub_id": sub_id_list}

	return data

class BSCdataset(Dataset):
	"""Return BSC dataset."""

	def __init__(
		self,
		word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer
	):
		self.data = _process_BSC_corpus(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf)

	def __len__(self):
		return len(self.data["SN_input_ids"])

	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

		sample["sub_id"] = self.data["sub_id"][idx]

		return sample


def calculate_mean_std(dataloader, feat_key, padding_value=0, scale=1):
	#calculate mean
	total_sum = 0
	total_num = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		total_num += len(feat.view(-1).nonzero())
		total_sum += feat.sum()
	feat_mean = total_sum / total_num
	#calculate std
	sum_of_squared_error = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		mask = ~torch.eq(feat, padding_value)
		sum_of_squared_error += (((feat - feat_mean).pow(2))*mask).sum()
	feat_std = torch.sqrt(sum_of_squared_error / total_num)
	return feat_mean, feat_std


def load_label(sp_pos, cf, labelencoder, device):
	#prepare label and mask
	pad_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"])
	end_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"]-1)
	mask = pad_mask + end_mask
	sac_amp = sp_pos[:, 1:] - sp_pos[:, :-1]
	label = sp_pos[:, 1:]*mask + sac_amp*~mask
	label = torch.where(label>cf["max_sn_len"]-1, cf["max_sn_len"]-1, label).to('cpu').detach().numpy()
	label = labelencoder.transform(label.reshape(-1)).reshape(label.shape[0], label.shape[1])
	label = torch.from_numpy(label).to(device)
	return pad_mask, label


def likelihood(pred, label, mask):
	#test
	#res = F.nll_loss(torch.tensor(pred), torch.tensor(label))
	mask = mask.cpu().numpy()
	label = one_hot_encode(label, pred.shape[1])
	res = np.sum(np.multiply(pred, label), axis=1)
	res = np.sum(res * ~mask)/np.sum(~mask)
	return res


def eval_log_llh(dnn_out, label, pad_mask):
	res = []
	dnn_out = np.log2(dnn_out + 1e-10)
	#For each scanpath calculate the likelihood and then find the average
	for sp_indx in range(dnn_out.shape[0]):
		out = likelihood(dnn_out[sp_indx, :, :], label[sp_indx, :], pad_mask[sp_indx, :])
		res.append(out)

	return res


def prepare_scanpath(sp_dnn, sn_len, sp_human, cf):
	max_sp_len = sp_dnn.shape[1]
	sp_human = sp_human.detach().to('cpu').numpy()

	#stop_indx = [np.where(sp_dnn[i,:]==(sn_len[i]+1))[0][0] for i in range(sp_dnn.shape[0])]
	#Find the number "sn_len+1" -> the end point
	stop_indx = []
	for i in range(sp_dnn.shape[0]):
		stop = np.where(sp_dnn[i,:]==(sn_len[i]+1))[0]
		if len(stop)==0:#no end point can be find -> exceeds the maximum length of the generated scanpath
			stop_indx.append(max_sp_len-1)
		else:
			stop_indx.append(stop[0])

	#Truncating data after the end point
	sp_dnn_cut = [sp_dnn[i][:stop_indx[i]+1] for i in range(sp_dnn.shape[0])]
	#replace the last teminal number to cf["max_sn_len"]-1, keep the same as the human scanpath label
	for i in range(len(sp_dnn_cut)):
		sp_dnn_cut[i][-1] = cf["max_sn_len"]-1

	#process the human scanpath data, truncating data after the end point
	stop_indx = [np.where(sp_human[i,:]==cf["max_sn_len"]-1)[0][0] for i in range(sp_human.shape[0])]
	sp_human_cut = [sp_human[i][:stop_indx[i]+1] for i in range(sp_human.shape[0])]
	return sp_dnn_cut, sp_human_cut


def celer_load_native_speaker():
	sub_metadata_path = './Data/celer/metadata.tsv'
	sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
	native_sub_list = sub_infor[sub_infor.L1 == 'English'].List.values
	return native_sub_list.tolist()


def compute_word_length_celer(arr):
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr = arr.astype('float64')
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr


def _process_celer(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf):
	"""
	SN_token_embedding   <CLS>, bla, bla, <SEP>
	SP_token_embedding       <CLS>, bla, bla, <SEP>
	SP_ordinal_pos 0, bla, bla, max_sp_len
	SP_fix_dur     0, bla, bla, 0
	Concatenate: concatenate scanpaths and sentences for training
	"""
	SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
	SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list, raw_split_words =  [], []
	for sn_id in tqdm(sn_list):
	 
		#process sentence sequence
		sn_df = eyemovement_df[eyemovement_df.sentenceid==sn_id]
		#notice: Each sentence is recorded multiple times in file |word_info_df|.
		sn = word_info_df[word_info_df.sentenceid == sn_id]
		sn = sn[sn['list']==sn.list.values.tolist()[0]]
		#compute word length for each word
		sn_word_len = compute_word_length_celer(sn.WORD_LEN.values)

		sn_str = sn.sentence.iloc[-1]
		#nessacery sanity check, when split sentence to words, the length of sentence should match the sentence length recorded in celer dataset
		if sn_id in ['1987/w7_019/w7_019.295-3', '1987/w7_036/w7_036.147-43', '1987/w7_091/w7_091.360-6']:
			#extra inverted commas at the end of the sentence
			sn_str = sn_str[:-3] + sn_str[-1:]
		if sn_id == '1987/w7_085/w7_085.200-18':
			sn_str = sn_str[:43] + sn_str[44:]
		sn_len = len(sn_str.split())

		#tokenization and padding
		tokenizer.padding_side = 'right'
  
		raw_split_text = sn_str.split()
		sn_str = '[CLS]' + ' ' + sn_str + ' ' + '[SEP]'
		#pre-tokenized input
		tokens = tokenizer.encode_plus(sn_str.split(),
										add_special_tokens = False,
										truncation=False,
										max_length = cf['max_sn_token'],
										padding = 'max_length',
										return_attention_mask=True,
										is_split_into_words=True)
		encoded_sn = tokens['input_ids']
		mask_sn = tokens['attention_mask']
		#use offset mapping to determine if two tokens are in the same word.
		#index start from 0, CLS -> 0 and SEP -> last index
		word_ids_sn = tokens.word_ids()
		word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

		#process fixation sequence
		for sub_id in reader_list:
			sub_df = sn_df[sn_df.list==sub_id]
			# remove fixations on non-words
			sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
			if len(sub_df) == 0:
				#no scanpath data found for the subject
				continue

			#prepare decoder input and output
			sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

			#dataset is noisy -> sanity check
			# 1) check if recorded fixation duration are within reasonable limits
			#Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
			outlier_indx = np.where(sp_fix_dur<50)[0]
			if outlier_indx.size>0:
				for out_idx in range(len(outlier_indx)):
					outlier_i = outlier_indx[out_idx]
					merge_flag = False

					#outliers are commonly found in the fixation of the last record and the first record, and are removed directly
					if outlier_i == len(sp_fix_dur)-1 or outlier_i == 0:
						merge_flag = True

					else:
						if outlier_i-1 >= 0 and merge_flag == False:
							#try to merge with the left fixation
							if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i-1].CURRENT_FIX_INTEREST_AREA_LABEL:
								sp_fix_dur[outlier_i-1] = sp_fix_dur[outlier_i-1] + sp_fix_dur[outlier_i]
								merge_flag = True

						if outlier_i+1 < len(sp_fix_dur) and merge_flag == False:
							#try to merge with the right fixation
							if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i+1].CURRENT_FIX_INTEREST_AREA_LABEL:
								sp_fix_dur[outlier_i+1] = sp_fix_dur[outlier_i+1] + sp_fix_dur[outlier_i]
								merge_flag = True

					sp_word_pos = np.delete(sp_word_pos, outlier_i)
					sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
					sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
					sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
					outlier_indx = outlier_indx-1

			# 2) scanpath too long, remove outliers, speed up the inference
			if len(sp_word_pos) > 50: # 72/10684
				continue
			# 3)scanpath too short for a normal length sentence
			if len(sp_word_pos)<=1 and sn_len>10:
				continue

			# 4) check landing position feature
			#assign missing value to 'nan'
			sp_fix_loc=np.where(sp_fix_loc=='.', np.nan, sp_fix_loc)
			#convert string of number of float type
			sp_fix_loc = [float(i) for i in sp_fix_loc]

			#Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
			if np.nanmax(sp_fix_loc)>35:
				missing_idx = np.where(np.array(sp_fix_loc)>5)[0]
				for miss in missing_idx:
					if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
						sp_fix_loc[miss] = np.nan
					else:
						print('Landing position calculation error. Unknown cause, needs to be checked')

			sp_ordinal_pos = sp_word_pos.astype(int)
			SP_ordinal_pos.append(sp_ordinal_pos)
			SP_fix_dur.append(sp_fix_dur)
			SP_landing_pos.append(sp_fix_loc)

			sp_token = [sn_str.split()[int(i)] for i in sp_ordinal_pos]
			sp_token_str = '[CLS]' + ' ' + ' '.join(sp_token) + ' ' + '[SEP]'

			#tokenization and padding for scanpath, i.e. fixated word sequence
			sp_tokens = tokenizer.encode_plus(sp_token_str.split(),
											add_special_tokens = False,
											truncation=False,
											max_length = cf['max_sp_token'],
											padding = 'max_length',
											return_attention_mask=True,
											is_split_into_words=True)
			encoded_sp = sp_tokens['input_ids']
			mask_sp = sp_tokens['attention_mask']
			#index start from 0, CLS -> 0 and SEP -> last index
			word_ids_sp = sp_tokens.word_ids()
   
   
			word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)
			WORD_ids_sp.append(word_ids_sp)

			#sentence information
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(sn_word_len)
			WORD_ids_sn.append(word_ids_sn)
			sub_id_list.append(int(sub_id))
   
			raw_split_words.append(raw_split_text)
			#print(SP_ordinal_pos, raw_split_words)

	raw_sp_pos = SP_ordinal_pos
 
	#padding for batch computation
	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)
 
	

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list, dtype=np.int64)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)


	data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len, "WORD_ids_sn": WORD_ids_sn,
	 		"SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask, "WORD_ids_sp": WORD_ids_sp,
			"SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
			"sub_id": sub_id_list, "raw_split_words": raw_split_words, "raw_sp_pos": raw_sp_pos
			}

	return data

class celerdataset(Dataset):
	"""Return celer dataset."""

	def __init__(
		self,
		word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer
	):
		self.data = _process_celer(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf)

	def __len__(self):
		return len(self.data["SN_input_ids"])


	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

		sample["sub_id"] = self.data["sub_id"][idx]

		return sample


def one_hot_encode(arr, dim):
	# one hot encode
	onehot_encoded = np.zeros((arr.shape[0], dim))
	for idx, value in enumerate(arr):
		onehot_encoded[idx, value] = 1

	return onehot_encoded

def gradient_clipping(dnn_model, clip = 10):
	torch.nn.utils.clip_grad_norm_(dnn_model.parameters(),clip)

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


def _process_meco(sn_df, data_df, cf, reader_list, sn_list, tokenizer):
	# Initialize lists to store processed data
	SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
	SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list, raw_split_words = [], []
	# We need to iterate over the subjects and then over the trialid or itemid
	for sub_id in tqdm(reader_list):
		lang_code = sub_id.split('_')[0]
		#if lang_code not in ['du', 'en', 'ee', 'fi', 'ge', 'it', 'no', 'es', 'tr']:
		#if lang_code not in ["en"]:
		#	continue
		# Iterate over sentences in the MeCo dataset
		for sent_id in sn_list:
			# Extract text by language rowID and column textID
			text = sn_df.loc[sn_df['Language'] == lang_code, sent_id].iloc[0].replace('"', '')
			#print(sent_id)
			#print(text[:100])
			subj_fix_df = data_df[(data_df['uniform_id'] == sub_id)\
       			& (data_df['itemid'] == sent_id)]
			if len(subj_fix_df) == 0:
				continue
			# Processing sentence sequence
			text_word_len = [len(word) for word in text.split()]  # Assuming word length calculation is similar

			# Tokenization and padding for sentence
			text_str = '[CLS] ' + text + ' [SEP]'
			tokens = tokenizer.encode_plus(text_str.split(),
										add_special_tokens=False,
										truncation=True, # Worst case truncate
										max_length=cf['max_sn_token'],
										padding='max_length',
										return_attention_mask=True,
										is_split_into_words=True)
   
			encoded_sn = tokens['input_ids']
			mask_sn = tokens['attention_mask']
			word_ids_sn = tokens.word_ids()
			word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

			# Process fixation sequence for each subject using uniform_id instead of subid
			# Extract relevant columns for fixation processing
			sp_word_pos = subj_fix_df['wordnum'].dropna().values
			sp_fix_dur = subj_fix_df['dur'].dropna().values
			sp_landing_pos = subj_fix_df['word.land'].dropna().values
			sp_words = subj_fix_df['word'].dropna().values.tolist()
			#print("word: ", sp_words[:20])
			split_text = re.split(r'(?<=-)\b|\s+', text_str)
			raw_slit_text = re.split(r'(?<=-)\b|\s+', text)
			# Processing for scanpath (fixated word sequence)
			try:
				sp_token = [split_text[int(i)] for i in sp_word_pos]
			except IndexError as e:
				print(f"Index Error: {e}")
				continue
   
			try:
				assert sp_words == sp_token
			except AssertionError as e: # Lists not completely equal, check for unequal words
				print(f"Assertion Error: {e}")
				print("subject id ", sub_id)
				with open("unequal_words3.txt", "a") as file:  # Open the file in append mode
					unequal_words = [(word1, word2) for word1, word2 in zip(sp_words, sp_token) if word1.strip('":-,.–;”∙').strip("'") != word2.strip('":-,.–;”∙').strip("'")]
					#print(unequal_words)
					# Write to file if there are unequal words
					if unequal_words: # If there are unequal words the index is shifted and we skip the sentence
						file.write(f"Original words: {sp_words}, Count: {len(sp_words)}\n")
						file.write(f"Extracted words: {sp_token}, Count: {len(sp_token)}\n")
						file.write(f"Unequal words: {unequal_words}\n\n")
						continue  # Continue with the next sentence of the loop
					
					if len(sp_words) != len(sp_token):
						print("Unequal length")
						continue  # Continue with the next iteration of the loop

			sp_token_str = '[CLS] ' + ' '.join(sp_token) + ' [SEP]'

			# Tokenization and padding for scanpath
			sp_tokens = tokenizer.encode_plus(sp_token_str.split(),
											add_special_tokens=False,
											truncation=True, # Worst case truncate
											max_length=cf['max_sp_token'],
											padding='max_length',
											return_attention_mask=True,
											is_split_into_words=True)
			encoded_sp = sp_tokens['input_ids']
			mask_sp = sp_tokens['attention_mask']
			word_ids_sp = sp_tokens.word_ids()
			word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
			if len(sp_landing_pos) > cf['max_sp_len']:
				print("Scanpath length exceeds max length")
				continue
			elif len(word_ids_sp) > cf['max_sp_token']:
				print("Scanpath token length exceeds max length")
				continue
			elif len(word_ids_sn) > cf['max_sn_token']:
				print("Sentence length exceeds max length")
				continue
			elif len(word_ids_sn) > cf['max_sn_token']:
				print("Sentence token length exceeds max length")
				continue
			# Append processed data to lists
			SP_ordinal_pos.append(sp_word_pos)
			SP_fix_dur.append(sp_fix_dur)
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)
			WORD_ids_sp.append(word_ids_sp)
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(text_word_len)
			WORD_ids_sn.append(word_ids_sn)
			sub_id_list.append(sub_id)
			SP_landing_pos.append(sp_landing_pos)

			raw_split_words.append(raw_slit_text)

	# Padding for batch computation 
	raw_sp_pos = SP_ordinal_pos
 
	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
 
	sub_id_list = np.asarray(sub_id_list)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)
 
	raw_split_words = raw_split_words

	# Create the final data structure
	data = {
		"SN_input_ids": np.array(SN_input_ids),
		"SN_attention_mask": np.array(SN_attention_mask),
		"SN_WORD_len": np.array(SN_WORD_len),
		"WORD_ids_sn": np.array(WORD_ids_sn),
		"SP_input_ids": np.array(SP_input_ids),
		"SP_attention_mask": np.array(SP_attention_mask),
		"WORD_ids_sp": np.array(WORD_ids_sp),
		"SP_ordinal_pos": np.array(SP_ordinal_pos),
		"SP_fix_dur": np.array(SP_fix_dur),
		"sub_id": np.array(sub_id_list),
		"SP_landing_pos": np.array(SP_landing_pos),
		"raw_split_words": raw_split_words,
		"raw_sp_pos": raw_sp_pos
	}

	return data

class mecodataset(Dataset):
	"""Return celer dataset."""
	def __init__(self, sn_df, data_df, cf, reader_list, sn_list, tokenizer):
		self.data = _process_meco(sn_df, data_df, cf, reader_list, sn_list, tokenizer)

	def __len__(self):
		return len(self.data["SN_input_ids"])


	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]
  
		sample["sub_id"] = self.data["sub_id"][idx]

		return sample


def merge_and_extract_datasets(dataset1, dataset2):
	merged_data = {}
	keys = ['raw_split_words', 'raw_sp_pos']
	dataset1 = dataset1.data
	dataset2 = dataset2.data
	# Select a key to generate shuffled indices based on its length
	key_to_shuffle = keys[0]
	# Determine the total length after concatenation
	total_length = len(dataset1[key_to_shuffle]) + len(dataset2[key_to_shuffle])
	# Generate shuffled indices
	shuffled_indices = list(range(total_length))
	import random
	random.shuffle(shuffled_indices)
	# Concatenate and then shuffle the lists from both datasets for each key
	for key in keys:
		concatenated_list = dataset1[key] + dataset2[key]
		# Apply the same shuffled indices to each concatenated list
		merged_data[key] = [concatenated_list[i] for i in shuffled_indices]
	return merged_data


def merge_and_shuffle_datasets(dataset1, dataset2):
    merged_data = {}
    # Select a key to generate shuffled indices based on its length
    key_to_shuffle = list(dataset1.data.keys())[0]
    # Determine the total length after concatenation
    total_length = dataset1.data[key_to_shuffle].shape[0] + dataset2.data[key_to_shuffle].shape[0]
    # Generate shuffled indices
    shuffled_indices = np.random.permutation(total_length)
    # Concatenate and then shuffle the arrays from both datasets for each key
    for key in dataset1.data:
        concatenated_array = np.concatenate((dataset1.data[key], dataset2.data[key]), axis=0)
        # Apply the same shuffled indices to each concatenated array
        merged_data[key] = concatenated_array[shuffled_indices]  
    return merged_data


def load_celer(path, split):
    return load_with_pickle(f'{path}celer_dataset_{split}_local-g.pickle')


def load_meco(path, split):
    return load_with_pickle(f'{path}meco_dataset_{split}_local-g.pickle')


def load_combined_dataset(path, split):
	celer = load_celer(path, split)
	meco = load_meco(path, split)
	return merge_and_shuffle_datasets(celer, meco)


def load_dataset(dataset_name, dataset_type, dataset_path):
	if dataset_name == "meco":
		return load_meco(dataset_path, dataset_type)
	elif dataset_name == "celer":
		return load_celer(dataset_path, dataset_type)
	elif dataset_name == "combined":
		return combineddataset(dataset_path, dataset_type)
	else:
		raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_and_load(cf):
	# Preprocess data corpus
	print("Preprocessing data corpus")
	if cf["dataset"] == 'meco':
		data_df, sn_df, reader_list = load_corpus(cf["dataset"])
		split_list = [col for col in sn_df.columns if col != 'Language']
	elif cf["dataset"] == 'celer':
		word_info_df, _, eyemovement_df = load_corpus(cf["dataset"])
		reader_list = celer_load_native_speaker()
		split_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

	initial_train, list_test = train_test_split(split_list, test_size=0.20, random_state=0)
	# Further splitting the training set into train and validation sets
	list_train, list_val = train_test_split(initial_train, test_size=0.15, random_state=0)
	reader_list_train, reader_list_val, reader_list_test = reader_list, reader_list, reader_list
	#initialize tokenizer
	tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])
	# Preparing batch data
	if cf["dataset"] == 'meco':
		return mecodataset(sn_df, data_df, cf, reader_list_train, list_train, tokenizer), \
			mecodataset(sn_df, data_df, cf, reader_list_val, list_val, tokenizer), \
			mecodataset(sn_df, data_df, cf, reader_list_test, list_test, tokenizer)
	elif cf["dataset"] == 'celer':
		return celerdataset(word_info_df, eyemovement_df, cf, reader_list_train, list_train, tokenizer), \
			celerdataset(word_info_df, eyemovement_df, cf, reader_list_val, list_val, tokenizer), \
			celerdataset(word_info_df, eyemovement_df, cf, reader_list_test, list_test, tokenizer)


class combineddataset(Dataset):
	"""Return celer dataset."""
	def __init__(self, path, split):
		self.data = load_combined_dataset(path, split)

	def __len__(self):
		return len(self.data["SN_input_ids"])

	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]
  
		sample["sub_id"] = self.data["sub_id"][idx]

		return sample
