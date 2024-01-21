import numpy as np
import os
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from collections import deque
import pickle
import argparse
from sklearn.model_selection import train_test_split

from model import Eyettention
from utils import load_pretrained_model, save_with_pickle

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run Eyettention on Meco dataset')
	parser.add_argument(
		'--atten_type',
		help='attention type: global, local, local-g',
		type=str,
		default='local-g'
	)
	parser.add_argument(
		'--save_data_folder',
		help='folder path for saving results',
		type=str,
		default='./results/meco/'
	)
	parser.add_argument(
		'--scanpath_gen_flag',
		help='whether to generate scanpath',
		type=int,
		default=1
	)
	parser.add_argument(
		'--max_pred_len',
		help='if scanpath_gen_flag is True, you can determine the longest scanpath that you want to generate, which should depend on the sentence length',
		type=int,
		default=256
	)
	parser.add_argument(
		'--gpu',
		help='gpu index',
		type=int,
		default=0
	)
	parser.add_argument(
		'--load_pretrained',
		help='load pretrained model',
		type=bool,
		default=False
	)
	parser.add_argument(
		'--pretrained_model_path',
		help='pretrained model path',
		type=str,
		default="results/meco/CELoss_meco_text_eyettention_local-g_newloss.pth"
	)
	parser.add_argument(
		'--load_dataset',
		help='load extracted dataset',
		type=bool,
		default=False
	)
	args = parser.parse_args()
	gpu = args.gpu

	#use FastTokenizer lead to warning -> The current process just got forked
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	torch.set_default_tensor_type('torch.FloatTensor')
	availbl = torch.cuda.is_available()
	print(torch.cuda.is_available())
	if availbl:
		device = f'cuda:{gpu}'
		torch.cuda.set_device(gpu)
	else:
		device = 'cpu'
	print(device)

	cf = {"model_pretrained": "bert-base-multilingual-cased", # Multiling BERT
			"lr": 1e-3,
			"max_grad_norm": 10,
			"n_epochs": 1000,
			"dataset": 'combined',
			"atten_type": args.atten_type,
			"batch_size": 32,
			"max_sn_len": 256,
			"max_sn_token": 400,
			"max_sp_len": 512,
			"max_sp_token": 512,
			"norm_type": 'z-score',
			"earlystop_patience": 20,
			"max_pred_len":args.max_pred_len
			}

	#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"]-1), cf["max_sn_len"]-1))
	#le.classes_
	if cf["dataset"] == "combined":
		# Loading combined dataset
		print("Loading combined dataset")
		dataset_train = combineddataset("train")
		dataset_val = combineddataset("val")
		dataset_test = combineddataset("test")
	else:
		# Preprocess data corpus
		print("Preprocessing data corpus")
		if cf["dataset"] == 'meco':
			data_df, sn_df, reader_list = load_corpus(cf["dataset"])
			split_list = filtered_columns = [col for col in sn_df.columns if col != 'Language']
	
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
		#Preparing batch data
		if cf["dataset"] == 'meco':
			dataset_train = mecodataset(sn_df, data_df, cf, reader_list_train, list_train, tokenizer)
			dataset_val = mecodataset(sn_df, data_df, cf, reader_list_val, list_val, tokenizer)
			dataset_test = mecodataset(sn_df, data_df, cf, reader_list_test, list_test, tokenizer)

		elif cf["dataset"] == 'celer':
			dataset_train = celerdataset(word_info_df, eyemovement_df, cf, reader_list_train, list_train, tokenizer)
			dataset_val = celerdataset(word_info_df, eyemovement_df, cf, reader_list_val, list_val, tokenizer)
			dataset_test = celerdataset(word_info_df, eyemovement_df, cf, reader_list_test, list_test, tokenizer)

		save_with_pickle(dataset_train, os.path.join(args.save_data_folder, f'{cf["dataset"]}_dataset_train_{args.atten_type}.pickle'))
		save_with_pickle(dataset_val, os.path.join(args.save_data_folder, f'{cf["dataset"]}_dataset_val_{args.atten_type}.pickle'))
		save_with_pickle(dataset_test, os.path.join(args.save_data_folder, f'{cf["dataset"]}_dataset_test_{args.atten_type}.pickle'))


	train_dataloaderr = DataLoader(dataset_train, batch_size = cf["batch_size"], shuffle = True, drop_last=True)
	val_dataloaderr = DataLoader(dataset_val, batch_size = cf["batch_size"], shuffle = False, drop_last=True)
	test_dataloaderr = DataLoader(dataset_test, batch_size = cf["batch_size"], shuffle = False, drop_last=False)

	#for scanpath generation
	loss_dict = {'val_loss':[], 'train_loss':[], 'test_ll':[], 'test_AUC':[]}
	sp_dnn_list = []
	sp_human_list = []
	
	#z-score normalization for gaze features
	fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_fix_dur", padding_value=0, scale=1000)
	landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_landing_pos", padding_value=0)
	sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

	# load model
	if args.load_pretrained:
		print("Loading pretrained")
		dnn = load_pretrained_model(args.pretrained_model_path, cf, device)
	else:
		print("Creating new model")
		dnn = Eyettention(cf)

	#training
	episode = 0
	optimizer = Adam(dnn.parameters(), lr=cf["lr"])
	dnn.train()
	dnn.to(device)
	av_score = deque(maxlen=100)
	old_score = 1e10
	save_ep_couter = 0
	print('Start training')
	for episode_i in range(episode, cf["n_epochs"]+1):
		dnn.train()
		print('episode:', episode_i)
		counter = 0
		for batchh in train_dataloaderr:
			counter += 1
			batchh.keys()
			sn_input_ids = batchh["sn_input_ids"].to(device)
			sn_attention_mask = batchh["sn_attention_mask"].to(device)
			word_ids_sn = batchh["word_ids_sn"].to(device)
			sn_word_len = batchh["sn_word_len"].to(device)

			sp_input_ids = batchh["sp_input_ids"].to(device)
			sp_attention_mask = batchh["sp_attention_mask"].to(device)
			word_ids_sp = batchh["word_ids_sp"].to(device)

			sp_pos = batchh["sp_pos"].to(device)
			sp_landing_pos = batchh["sp_landing_pos"].to(device)
			sp_fix_dur = (batchh["sp_fix_dur"]/1000).to(device)

			#normalize gaze features
			mask = ~torch.eq(sp_fix_dur, 0)
			sp_fix_dur = (sp_fix_dur-fix_dur_mean)/fix_dur_std * mask
			sp_landing_pos = (sp_landing_pos - landing_pos_mean)/landing_pos_std * mask
			sp_fix_dur = torch.nan_to_num(sp_fix_dur)
			sp_landing_pos = torch.nan_to_num(sp_landing_pos)
			sn_word_len = (sn_word_len - sn_word_len_mean)/sn_word_len_std
			sn_word_len = torch.nan_to_num(sn_word_len)

			# zero old gradients
			optimizer.zero_grad()
			# predict output with DNN
			dnn_out, atten_weights = dnn(sn_emd=sn_input_ids,
										sn_mask=sn_attention_mask,
										sp_emd=sp_input_ids,
										sp_pos=sp_pos,
										word_ids_sn=word_ids_sn,
										word_ids_sp=word_ids_sp,
										sp_fix_dur=sp_fix_dur,
										sp_landing_pos=sp_landing_pos,
										sn_word_len = sn_word_len)

			dnn_out = dnn_out.permute(0,2,1)              #[batch, dec_o_dim, step]

			#prepare label and mask
			pad_mask, label = load_label(sp_pos, cf, le, device)
			loss = nn.CrossEntropyLoss(reduction="none")
			batch_error = torch.mean(torch.masked_select(loss(dnn_out, label), ~pad_mask))

			# backpropagate loss
			batch_error.backward()
			# clip gradients
			gradient_clipping(dnn, cf["max_grad_norm"])

			#learn
			optimizer.step()
			av_score.append(batch_error.to('cpu').detach().numpy())
			print('counter:',counter)
			print('\rSample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")
		loss_dict['train_loss'].append(np.mean(av_score))

		val_loss = []
		dnn.eval()
		for batchh in val_dataloaderr:
			with torch.no_grad():
				sn_input_ids_val = batchh["sn_input_ids"].to(device)
				sn_attention_mask_val = batchh["sn_attention_mask"].to(device)
				word_ids_sn_val = batchh["word_ids_sn"].to(device)
				sn_word_len_val = batchh["sn_word_len"].to(device)

				sp_input_ids_val = batchh["sp_input_ids"].to(device)
				sp_attention_mask_val = batchh["sp_attention_mask"].to(device)
				word_ids_sp_val = batchh["word_ids_sp"].to(device)

				sp_pos_val = batchh["sp_pos"].to(device)
				sp_landing_pos_val = batchh["sp_landing_pos"].to(device)
				sp_fix_dur_val = (batchh["sp_fix_dur"]/1000).to(device)

				#normalize gaze features
				mask = ~torch.eq(sp_fix_dur_val, 0)
				sp_fix_dur_val = (sp_fix_dur_val-fix_dur_mean)/fix_dur_std * mask
				sp_landing_pos_val = (sp_landing_pos_val - landing_pos_mean)/landing_pos_std * mask
				sp_fix_dur_val = torch.nan_to_num(sp_fix_dur_val)
				sp_landing_pos_val = torch.nan_to_num(sp_landing_pos_val)
				sn_word_len_val = (sn_word_len_val - sn_word_len_mean)/sn_word_len_std
				sn_word_len_val = torch.nan_to_num(sn_word_len_val)

				dnn_out_val, atten_weights_val = dnn(sn_emd=sn_input_ids_val,
													sn_mask=sn_attention_mask_val,
													sp_emd=sp_input_ids_val,
													sp_pos=sp_pos_val,
													word_ids_sn=word_ids_sn_val,
													word_ids_sp=word_ids_sp_val,
													sp_fix_dur=sp_fix_dur_val,
													sp_landing_pos=sp_landing_pos_val,
													sn_word_len = sn_word_len_val)
				dnn_out_val = dnn_out_val.permute(0,2,1)              #[batch, dec_o_dim, step

				#prepare label and mask
				pad_mask_val, label_val = load_label(sp_pos_val, cf, le, device)
				batch_error_val = torch.mean(torch.masked_select(loss(dnn_out_val, label_val), ~pad_mask_val))
				val_loss.append(batch_error_val.detach().to('cpu').numpy())
		print('\nvalidation loss is {} \n'.format(np.mean(val_loss)))
		loss_dict['val_loss'].append(np.mean(val_loss))

		if np.mean(val_loss) < old_score:
			# save model if val loss is smallest
			torch.save(dnn.state_dict(), f'{args.save_data_folder}/CELoss_{cf["dataset"]}_eyettention_{args.atten_type}_newloss.pth')
			old_score= np.mean(val_loss)
			print('\nsaved model state dict\n')
			save_ep_couter = episode_i
		else:
			#early stopping
			if episode_i - save_ep_couter >= cf["earlystop_patience"]:
				break

	#evaluation
	dnn.eval()
	res_llh=[]
	dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,\
     	f'CELoss_{cf["dataset"]}_{args.test_mode}_eyettention_{args.atten_type}_newloss.pth'), map_location='cpu'))
	dnn.to(device)
	batch_indx = 0
	for batchh in test_dataloaderr:
		with torch.no_grad():
			sn_input_ids_test = batchh["sn_input_ids"].to(device)
			sn_attention_mask_test = batchh["sn_attention_mask"].to(device)
			word_ids_sn_test = batchh["word_ids_sn"].to(device)
			sn_word_len_test = batchh["sn_word_len"].to(device)

			sp_input_ids_test = batchh["sp_input_ids"].to(device)
			sp_attention_mask_test = batchh["sp_attention_mask"].to(device)
			word_ids_sp_test = batchh["word_ids_sp"].to(device)

			sp_pos_test = batchh["sp_pos"].to(device)
			sp_landing_pos_test = batchh["sp_landing_pos"].to(device)
			sp_fix_dur_test = (batchh["sp_fix_dur"]/1000).to(device)

			#normalize gaze features
			mask = ~torch.eq(sp_fix_dur_test, 0)
			sp_fix_dur_test = (sp_fix_dur_test-fix_dur_mean)/fix_dur_std * mask
			sp_landing_pos_test = (sp_landing_pos_test - landing_pos_mean)/landing_pos_std * mask
			sp_fix_dur_test = torch.nan_to_num(sp_fix_dur_test)
			sp_landing_pos_test = torch.nan_to_num(sp_landing_pos_test)
			sn_word_len_test = (sn_word_len_test - sn_word_len_mean)/sn_word_len_std
			sn_word_len_test = torch.nan_to_num(sn_word_len_test)

			dnn_out_test, atten_weights_test = dnn(sn_emd=sn_input_ids_test,
													sn_mask=sn_attention_mask_test,
													sp_emd=sp_input_ids_test,
													sp_pos=sp_pos_test,
													word_ids_sn=word_ids_sn_test,
													word_ids_sp=word_ids_sp_test,
													sp_fix_dur=sp_fix_dur_test,
													sp_landing_pos=sp_landing_pos_test,
													sn_word_len = sn_word_len_test)

			#We do not use nn.CrossEntropyLoss here to calculate the likelihood because it combines nn.LogSoftmax and nn.NLL,
			#while nn.LogSoftmax returns a log value based on e, we want 2 instead
			#m = nn.LogSoftmax(dim=2) -- base e, we want base 2
			m = nn.Softmax(dim=2)
			dnn_out_test = m(dnn_out_test).detach().to('cpu').numpy()

			#prepare label and mask
			pad_mask_test, label_test = load_label(sp_pos_test, cf, le, 'cpu')
			pred = dnn_out_test.argmax(axis=2)
			#compute log likelihood for the batch samples
			res_batch = eval_log_llh(dnn_out_test, label_test, pad_mask_test)
			res_llh.append(np.array(res_batch))

			if bool(args.scanpath_gen_flag) == True:
				sn_len = (torch.max(torch.nan_to_num(word_ids_sn_test), dim=1)[0]+1-2).detach().to('cpu').numpy()
				#compute the scan path generated from the model when the first few fixed points are given
				sp_dnn = dnn.scanpath_generation(sn_emd=sn_input_ids_test,
													sn_mask=sn_attention_mask_test,
													word_ids_sn=word_ids_sn_test,
													sn_word_len = sn_word_len_test,
													le=le,
													max_pred_len=cf['max_pred_len'])

				sp_dnn, sp_human = prepare_scanpath(sp_dnn[0].detach().to('cpu').numpy(), sn_len, sp_pos_test, cf)
				sp_dnn_list.extend(sp_dnn)
				sp_human_list.extend(sp_human)

			batch_indx +=1

	res_llh = np.concatenate(res_llh).ravel()
	loss_dict['test_ll'].append(res_llh)
	loss_dict['fix_dur_mean'] = fix_dur_mean
	loss_dict['fix_dur_std'] = fix_dur_std
	loss_dict['landing_pos_mean'] = landing_pos_mean
	loss_dict['landing_pos_std'] = landing_pos_std
	loss_dict['sn_word_len_mean'] = sn_word_len_mean
	loss_dict['sn_word_len_std'] = sn_word_len_std
	print('\nTest likelihood is {} \n'.format(np.mean(res_llh)))
	#save results
	with open(f'{args.save_data_folder}/res_{cf["dataset"]}_eyettention_{args.atten_type}.pickle', 'wb') as handle:
		pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	if bool(args.scanpath_gen_flag) == True:
		#save results
		dic = {"sp_dnn": sp_dnn_list, "sp_human": sp_human_list}
		with open(f'{args.save_data_folder}/{cf["dataset"]}_scanpath_generation_eyettention_{args.test_mode}_{args.atten_type}.pickle', 'wb') as handle:
			pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
