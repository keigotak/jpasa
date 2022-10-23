import sys
sys.path.append('../')
import os
import numpy as np
import random
import gc
import copy
from pathlib import Path
import pickle
import json
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append(os.pardir)
from Datasets import get_datasets_in_sentences
# from utils.Vocab import Vocab
from Indexer import Indexer
from Validation import get_pr_numbers2, get_f_score

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

import transformers
transformers.BertTokenizer = transformers.BertJapaneseTokenizer
transformers.trainer_utils.set_seed(0)

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model
from transformers import T5Tokenizer, T5TokenizerFast, AutoModelForCausalLM, T5Model
from transformers import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import BertJapaneseTokenizer, BertModel
from BertJapaneseTokenizerFast import BertJapaneseTokenizerFast



class NTCDataset(Dataset):
    def __init__(self, items):
        super().__init__()
        self.tokens = items['tokens']
        self.labels = items['labels']
        self.properties = items['properties']
        self.word_positions = items['word_positions']
        self.ku_positions = items['ku_positions']
        self.modes = items['modes']
        self.flat_tokens = items['flat_tokens']
        self.preds = items['preds']
        self.pred_token_positions = items['pred_token_positions']
        self.input_tokens = items['input_tokens']
        self.sentences = items['sentences']
        self.original_tokens = items['original_tokens']
        self.tokenized_tokens = items['tokenized_tokens']
        self.pred_token_labels = items['pred_token_labels']

        self.merged_labels = items['merged_labels']
        self.merged_properties = items['merged_properties']
        self.merged_word_positions = items['merged_word_positions']
        self.merged_ku_positions = items['merged_ku_positions']
        self.merged_modes = items['merged_modes']

        self.length = len(self.input_tokens)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return {'input_tokens': self.input_tokens[index],
        'word_positions': self.merged_word_positions[index],
        'ku_positions': self.merged_ku_positions[index],
        'modes': self.merged_modes[index],
        'labels': self.merged_labels[index],
        'properties': self.merged_properties[index],
        'sentences': self.sentences[index],
        'preds': self.preds[index],
        'pred_token_positions': self.pred_token_positions[index],
        'pred_token_labels': self.pred_token_labels[index],
        'tokenized_tokens': self.tokenized_tokens[index]}


class SequencePointing(nn.Module):
    def __init__(self, hparams={}):
        super(SequencePointing, self).__init__()
        self.embedding_dim = hparams['token_embedding_dim'] + sum(list(hparams['other_embedding_dim'].values()))
        self.hidden_size = self.embedding_dim
        self.num_layers = hparams['num_layers']
        self.with_use_rnn_repeatedly = hparams['with_use_rnn_repeatedly']
        # self.vocab_size = hparams['vocab_size']

        if self.with_use_rnn_repeatedly:
            self.encoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
            self.decoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
            self.encoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
            self.decoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
        else:
            self.encoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])
            self.decoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])
            self.encoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])
            self.decoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])

        self.with_pn = False
        self.logit_types = ['g', 'w', 'n']
        self.label_size = len(self.logit_types) + 1
        # self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=hyper_parameters['num_heads'])
        self.linear_logits = nn.ModuleDict({key: nn.Linear(self.hidden_size, 1) for key in self.logit_types})
        # self.linear_logits = nn.Linear(self.hidden_size, self.label_size)
        # self.linear_decoder = nn.Linear(self.hidden_size, self.vocab_size)
        #
        # self.projection_matrix_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.projection_matrix_decoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.projection_vector = nn.Linear(self.hidden_size, 1, bias=False)
        #
        # self.decoder_start_input = nn.Parameter(torch.FloatTensor(self.embedding_dim))
        # self.decoder_start_input.data.uniform_(
        #     -(1. / np.sqrt(self.embedding_dim)), 1. / np.sqrt(self.embedding_dim)
        # )

    def forward(self, embeddings):
        x = embeddings.transpose(0, 1)

        for i in range(self.num_layers):
            for enc in [self.encoder_f[i], self.encoder_b[i]]:
                encoder_outputs, hidden = enc(x)
                x = encoder_outputs + x
                x = x[torch.arange(x.shape[0]-1, -1, -1), :, :]
        encoder_outputs = x.clone()

        # x, weights = self.attention(x, x, x)

        # for i in range(self.num_layers):
        #     for dec in [self.decoder_f[i], self.decoder_b[i]]:
        #         decoder_outputs, hidden = dec(x, hidden)
        #         x = decoder_outputs + x
        #         x = x[torch.arange(x.shape[0]-1, -1, -1), :, :]
        # decoder_outputs = x.clone()
        x = x.transpose(0, 1)

        logits = {t: self.linear_logits[t](x).squeeze(2) for t in self.logit_types}

        return {'logits': logits, 'last_hidden_state': x}

        # x_projected_encoder = self.projection_matrix_encoder(encoder_outputs)
        # x_projected_decoder = self.projection_matrix_decoder(decoder_outputs)
        # pointer_outputs = self.projection_vector(torch.selu(x_projected_encoder + x_projected_decoder))
        #
        # decoder_outputs = self.linear_decoder(x).transpose(1, 2)

        # return {'logits': logits, 'decoder_outputs': decoder_outputs, 'pointer_outputs': pointer_outputs}


class Embeddings(nn.Module):
    def __init__(self, hparams={}):
        super(Embeddings, self).__init__()
        self.device = hparams['DEVICE']
        self.model_name = hparams['model_name']
        if 'gpt2' in self.model_name:
            self.token_embeddings = AutoModel.from_pretrained(self.model_name)
            self.token_embedding_dim = self.token_embeddings.config.hidden_size
        elif 't5' in self.model_name:
            self.token_embeddings = T5Model.from_pretrained(self.model_name)
            self.token_embedding_dim = self.token_embeddings.config.d_model
        elif 'tohoku' in self.model_name:
            self.token_embeddings = BertModel.from_pretrained(self.model_name)
            self.token_embedding_dim = self.token_embeddings.config.hidden_size
        else:
            self.token_embeddings = AutoModel.from_pretrained(self.model_name)
            if self.model_name in set(['rinna/japanese-gpt-1b']):
                self.token_embedding_dim = self.model.embed_dim
            elif self.model_name in set(['rinna/japanese-roberta-base', 'nlp-waseda/roberta-base-japanese', 'nlp-waseda/roberta-large-japanese', 'xlm-roberta-large', 'xlm-roberta-base']):
                self.token_embedding_dim = self.token_embeddings.config.hidden_size
            else:
                self.token_embedding_dim = self.token_embeddings.config.d_model

        self.other_embedding_items = list(hparams['other_embedding_dim'].keys())
        self.other_embedding_dim = hparams['other_embedding_dim']
        self.other_embedding_vocab_size = hparams['other_embedding_vocab_size']
        self.other_embeddings = torch.nn.ModuleDict({k: torch.nn.Embedding(embedding_dim=self.other_embedding_dim[k], num_embeddings=self.other_embedding_vocab_size[k]) for k in self.other_embedding_items})

    def forward(self, tokens):
        x_token = self.token_embeddings(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        x_others = [self.other_embeddings[k](tokens[k]) for k in self.other_embedding_items]

        return torch.cat([x_token['last_hidden_state']] + x_others, dim=2)
    
    def get_token_embedding_dim(self):
        return self.token_embedding_dim



def get_aligned_dataset(args, preds, labels, props, word_pos, ku_pos, modes, indexers):
    sentences = [''.join(arg) for arg in args]
    input_tokens = [indexers['input_tokens'](sentence, return_tensors='pt') for sentence in sentences]
    original_tokens = [arg for arg in args]
    tokenized_tokens = [indexers['input_tokens'].convert_ids_to_tokens(token_input['input_ids'][0]) for token_input in input_tokens]

    # get flat sequences
    flat_tokens = [list(tokens) for tokens in sentences]
    flat_labels = [[l for a, l in zip(aa, ll) for _ in range(len(a))] for aa, ll in zip(args, labels)]
    flat_properties = [[l for a, l in zip(aa, ll) for _ in range(len(a))] for aa, ll in zip(args, props)]
    flat_word_positions = [[l for a, l in zip(aa, ll) for _ in range(len(a))] for aa, ll in zip(args, word_pos)]
    flat_ku_positions = [[l for a, l in zip(aa, ll) for _ in range(len(a))] for aa, ll in zip(args, ku_pos)]
    flat_modes = [[l for a, l in zip(aa, ll) for _ in range(len(a))] for aa, ll in zip(args, modes)]
    flat_pred_positions = [[item, item+len(pred[0])] for pred, item in zip(preds, [[l for a, l in zip(aa, ll) for _ in range(len(a))].index(2) for aa, ll in zip(args, word_pos)])]
    flat_preds = [''.join(flat_token[item: item+len(pred[0])]) for flat_token, pred, item in zip(flat_tokens, preds, [[l for a, l in zip(aa, ll) for _ in range(len(a))].index(2) for aa, ll in zip(args, word_pos)])]
    flat_items = {
        'tokens': flat_tokens,
        'labels': flat_labels,
        'properties': flat_properties,
        'word_positions': flat_word_positions,
        'ku_positions': flat_ku_positions,
        'modes': flat_modes,
        'pred_positions': flat_pred_positions,
        'preds': flat_preds
    }

    # exclude wrong dataset
    exclude_indexes = []
    for i, (_flat_pred, _pred) in enumerate(zip(flat_preds, preds)):
        _pred = _pred[0]
        if _flat_pred != _pred:
            print(f'{i}: {_flat_pred}, {_pred}')
            exclude_indexes.append(i)

    # align sequences
    item_types = ['tokens', 'labels', 'properties', 'word_positions', 'ku_positions', 'modes']
    batched_merged_items = {item_type: [] for item_type in item_types}
    for j, input_token in enumerate(input_tokens):
        if j in exclude_indexes:
            continue
        offsets = input_token['offset_mapping'] if 'offset_mapping' in input_token.keys() else input_token.encodings[0].offsets
        if offsets[0] != (0, 0) and offsets[1] == (0, 0):
            offsets[1] = (offsets[0][0], offsets[2][0])
            offsets[0] = (0, 0)
        if offsets[0] == (0, 1) and offsets[1][0] == 0:
            offsets[0] = (0, 0)
        if offsets[-1] != (0, 0) and offsets[-2] == (0, 0):
            offsets[-2] = (offsets[-3][1], offsets[-1][1])
            offsets[-1] = (0, 0)
        for item_type in item_types:
            merged_item = []
            for i, offset in enumerate(offsets):
                if len(offset) == 0:
                    merged_item.append(None)
                else:
                    merged_item.append(flat_items[item_type][j][offset[0]: offset[1]])
            batched_merged_items[item_type].append(merged_item.copy())

    batched_merged_items['flat_tokens'] = flat_tokens
    batched_merged_items['preds'] = flat_preds
    batched_merged_items['pred_token_positions'] = flat_pred_positions
    batched_merged_items['input_tokens'] = input_tokens
    batched_merged_items['sentences'] = sentences
    batched_merged_items['original_tokens'] = original_tokens
    batched_merged_items['tokenized_tokens'] = tokenized_tokens

    batched_pred_positions = []
    for input_token, pred_token_positions, pred, sentence in zip(batched_merged_items['input_tokens'], batched_merged_items['pred_token_positions'], batched_merged_items['preds'], batched_merged_items['sentences']):
        offsets = input_token['offset_mapping'] if 'offset_mapping' in input_token.keys() else input_token.encodings[0].offsets
        pred_positions = []
        for i, offset in enumerate(offsets):
            if offset[0] <= pred_token_positions[0]:
                if offset[1] <= pred_token_positions[0]:
                    pred_positions.append(0)
                elif pred_token_positions[0] < offset[1] <= pred_token_positions[1]:
                    pred_positions.append(1)
                elif offset[1] > pred_token_positions[1]:
                    pred_positions.append(1)
            elif pred_token_positions[0] < offset[0] <= pred_token_positions[1]:
                if offset[1] <= pred_token_positions[0]:
                    pred_positions.append(0)
                elif pred_token_positions[0] < offset[1] <= pred_token_positions[1]:
                    pred_positions.append(1)
                elif offset[1] > pred_token_positions[1]:
                    pred_positions.append(1)
            elif offset[0] > pred_token_positions[1]:
                if offset[1] <= pred_token_positions[0]:
                    pred_positions.append(0)
                elif pred_token_positions[0] < offset[1] <= pred_token_positions[1]:
                    pred_positions.append(0)
                elif offset[1] > pred_token_positions[1]:
                    pred_positions.append(0)
            else:
                pred_positions.append(-1)

        if 1 not in set(pred_positions):
            pred_positions[-1] = 1 # for dev error case
            print(f'pred not found: {i}')
        batched_pred_positions.append(pred_positions.copy())

    batched_merged_items['pred_token_labels'] = batched_pred_positions

    # FOR TEST
    with_test = False
    if with_test:
        for p, t, ft, l in zip(batched_merged_items['preds'], batched_merged_items['tokens'], batched_merged_items['flat_tokens'], batched_merged_items['pred_token_labels']):
            token = ''
            for it, il in zip(t, l):
                if il == 1:
                    token = token + ''.join(it)
            if p in token:
                pass
            else:
                print(f'{p} not in {token}: {"".join(ft)}')

    # merge labels
    merge_items = ['labels', 'properties', 'word_positions', 'ku_positions', 'modes']
    ignore_labels = [3, 'zero', 0, 0, 0]
    all_batched_new_items = {}
    for merge_item, ignore_label in zip(merge_items, ignore_labels):
        batched_new_items = []
        for _, items in enumerate(batched_merged_items[merge_item]):
            new_items = []
            for item in items:
                item = set(item)
                if len(item) == 0:
                    new_items.append(ignore_label)
                elif len(item) == 1:
                    new_items.append(list(item)[0])
                else:
                    if len(item - set([ignore_label])) == 0:
                        new_items.append(ignore_label)
                    elif len(item - set([ignore_label])) == 1:
                        new_items.append(list(item - set([ignore_label]))[0])
                    else:
                        item = item - set([ignore_label])
                        if merge_item in set(['labels', 'word_positions', 'ku_positions', 'modes']):
                            new_items.append(min(item))
                        elif merge_item in set(['properties']):
                            if 'pred' in item:
                                new_items.append('pred')
                            else:
                                new_items.append(list(item)[-1])
                                print(f'CONFLICT: {merge_item}, {item}')
                        else:
                            print(f'CONFLICT: {merge_item}, {item}')
            if len(items) != len(new_items):
                print(f'LENGTH ERROR: {items}, {new_items}')
            batched_new_items.append(new_items.copy())

        all_batched_new_items[merge_item] = batched_new_items.copy()

    batched_merged_items['merged_labels'] = all_batched_new_items['labels']
    batched_merged_items['merged_properties'] = all_batched_new_items['properties']
    batched_merged_items['merged_word_positions'] = indexers['word_positions'].transform_sentences(all_batched_new_items['word_positions']).tolist()
    batched_merged_items['merged_ku_positions'] = indexers['ku_positions'].transform_sentences(all_batched_new_items['ku_positions']).tolist()
    batched_merged_items['merged_modes'] = indexers['modes'].transform_sentences(all_batched_new_items['modes']).tolist()

    return batched_merged_items

def decoding_for_pointer(xs, batched_labels):
    _xs = {k: torch.cat([torch.zeros(v.shape[0], 1, device=v.device), v], dim=1) for k, v in xs.items()}
    _batched_labels = [[3] + labels for labels in batched_labels.tolist()]
    # _labels = torch.cat([torch.as_tensor([[3] for _ in range(batched_labels.shape[0])], device=batched_labels.device), batched_labels], dim=1)
    # mask_padding_positions = {k: v.masked_fill(_labels==4, -10000.0) for k, v in _xs.items()} # consideration of padding logits
    max_logits_positions = {k: torch.argmax(v, dim=1) for k, v in _xs.items()}
    _batched_positional_labels = {k: torch.as_tensor([labels.index(v) if v in set(labels) else 0 for labels in _batched_labels], dtype=torch.long, device=batched_labels.device) for k, v in {'g': 0, 'w': 1, 'n': 2}.items()}
    return {'logits': _xs, 'labels': _batched_labels, 'positional_labels': _batched_positional_labels, 'positional_predictions': max_logits_positions}

def decoding_for_inference(xs, sequence_length):
    one_hot_vectors = torch.stack([torch.nn.functional.one_hot(v, num_classes=sequence_length + 1) for v in xs.values()], dim=1).transpose(1, 2).tolist()
    predicted_labels = []
    for vv in one_hot_vectors:
        predicted_label = []
        for v in vv:
            unique_items = list(set(v) - set([0]))
            if len(unique_items) == 0:
                predicted_label.append(3)
            elif len(unique_items) == 1:
                predicted_label.append([i for i, item in enumerate(v) if item == 1][0])
            else:
                predicted_label.append([i for i, item in enumerate(v) if item == 1])
        predicted_labels.append(predicted_label)

    # predicted_labels = torch.as_tensor([[token_labels.index(1) if 1 in set(token_labels) else 3 for token_labels in sequence_labels] for sequence_labels in one_hot_vectors], device=xs['g'].device)
    return predicted_labels

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', './results/jpasa.rinna-japanese-gpt2-medium.pn', 30
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', './results/jpasa.bert-base-japanese-whole-word-masking.pn', 30
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', './results/jpasa.mbart-large-cc25.pn', 30
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', './results/jpasa.t5-base-japanese-web.pn', 30
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', './results/jpasa.rinna-japanese-roberta-base.pn', 30
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', './results/jpasa.nlp-waseda-roberta-base-japanese.pn', 30
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', './results/jpasa.nlp-waseda-roberta-large-japanese.pn', 30
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', './results/jpasa.rinna-japanese-gpt-1b.pn', 30
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', './results/jpasa.xlm-roberta-large.pn', 30
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', './results/jpasa.xlm-roberta-base.pn', 30

def allocate_data_to_device(data, device='cpu'):
    if device != 'cpu':
        return data.to('cuda:0')
    else:
        return data

def collate_fn(batch):
    max_sequence_length = max(map(len, [items['labels'] for items in batch]))
    padding_tokens = {'input_ids': PAD_ID_TOKEN, 'attention_mask': 0, 'word_positions': PAD_ID_WORD_POSITION, 'ku_positions': PAD_ID_KU_POSITION, 'modes': PAD_ID_MODE, 'labels': PAD_ID_LABEL}

    outputs = {}
    for k in ['input_ids', 'attention_mask']:
        batch_items = []
        for items in batch:
            items = items['input_tokens']
            seqence_length = items[k].shape[1]
            batch_items.append(torch.concat((items[k], torch.as_tensor([[padding_tokens[k]] * (max_sequence_length - seqence_length)])), 1) if seqence_length < max_sequence_length else items[k])
        outputs[k] = torch.stack(batch_items, dim=1).squeeze(0)

    for k in ['word_positions', 'ku_positions', 'modes', 'labels']:
        outputs[k] = torch.as_tensor([items[k] + [padding_tokens[k]] * (max_sequence_length - len(items[k])) if len(items[k]) < max_sequence_length else items[k] for items in batch])

    outputs['properties'] = [item['properties'] for item in batch]
    outputs['all'] = batch
    return outputs

def train(run_mode):
    global PAD_ID_TOKEN, PAD_ID_LABEL, PAD_ID_WORD_POSITION, PAD_ID_KU_POSITION, PAD_ID_MODE

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    TRAIN = "train2"
    DEV = "dev"
    TEST = "test"
    PAD_ID_LABEL = 4
    WITH_BERT = False
    WITH_BCCWJ = False

    TRAINING_BATCH_SIZE = 32
    INFERENCE_BATCH_SIZE = 256
    # WARMUP_STEPS = int(1000 // TRAINING_BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0'
    LR = 2e-6 if 'xlm' in run_mode else 2e-5
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = -1.0

    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.221022.0'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(TRAIN, with_bccwj=WITH_BCCWJ, with_bert=WITH_BERT)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(DEV, with_bccwj=WITH_BCCWJ, with_bert=WITH_BERT)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(TEST, with_bccwj=WITH_BCCWJ, with_bert=WITH_BERT)

    # vocab = Vocab()
    # vocab.fit(train_vocab)
    # vocab.fit(dev_vocab)
    # vocab.fit(test_vocab)
    # train_arg_id, train_pred_id = vocab.transform_sentences(train_args, train_preds)
    # dev_arg_id, dev_pred_id = vocab.transform_sentences(dev_args, dev_preds)
    # test_arg_id, test_pred_id = vocab.transform_sentences(test_args, test_preds)

    word_pos_indexer = Indexer()
    word_pos_id = np.concatenate([train_word_pos_id, dev_word_pos_id, test_word_pos_id])
    word_pos_indexer.fit(word_pos_id)
    train_word_pos = word_pos_indexer.transform_sentences(train_word_pos)
    dev_word_pos = word_pos_indexer.transform_sentences(dev_word_pos)
    test_word_pos = word_pos_indexer.transform_sentences(test_word_pos)
    PAD_ID_WORD_POSITION = word_pos_indexer.get_pad_id()

    ku_pos_indexer = Indexer()
    ku_pos_id = np.concatenate([train_ku_pos_id, dev_ku_pos_id, test_ku_pos_id])
    ku_pos_indexer.fit(ku_pos_id)
    train_ku_pos = ku_pos_indexer.transform_sentences(train_ku_pos)
    dev_ku_pos = ku_pos_indexer.transform_sentences(dev_ku_pos)
    test_ku_pos = ku_pos_indexer.transform_sentences(test_ku_pos)
    PAD_ID_KU_POSITION = ku_pos_indexer.get_pad_id()

    mode_indexer = Indexer()
    modes_id = np.concatenate([train_modes_id, dev_modes_id, test_modes_id])
    mode_indexer.fit(modes_id)
    train_modes = mode_indexer.transform_sentences(train_modes)
    dev_modes = mode_indexer.transform_sentences(dev_modes)
    test_modes = mode_indexer.transform_sentences(test_modes)
    PAD_ID_MODE = mode_indexer.get_pad_id()

    if 'gpt2' in model_name:
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    elif 't5' in model_name:
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
    elif 'tohoku' in model_name:
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    hyper_parameters = {
        'date': datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        'DEVICE': DEVICE,
        'model_name': model_name,
        'output_path': OUTPUT_PATH,
        'token_vocab_size': tokenizer.vocab_size,
        'with_use_rnn_repeatedly': False,
        'num_layers': 2,
        'num_heads': 1,
        'num_epochs': NUM_EPOCHS,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'GRADIENT_CLIP': GRADIENT_CLIP,
        'TRAINING_BATCH_SIZE': TRAINING_BATCH_SIZE,
        'INFERENCE_BATCH_SIZE': INFERENCE_BATCH_SIZE,
        'WITH_BERT': WITH_BERT,
        'WITH_BCCWJ': WITH_BCCWJ,
        'other_embedding_dim': {'word_positions': 20, 'ku_positions': 20, 'modes': 20},
        'other_embedding_vocab_size': {'word_positions': len(word_pos_indexer), 'ku_positions': len(ku_pos_indexer), 'modes': len(mode_indexer)}
    }

    embeddings = allocate_data_to_device(Embeddings(hparams=hyper_parameters), DEVICE)
    hyper_parameters['token_embedding_dim'] = embeddings.get_token_embedding_dim()
    decoder = allocate_data_to_device(SequencePointing(hparams=hyper_parameters), DEVICE)

    for k, v in embeddings.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))

    optimizer = torch.optim.AdamW(list(embeddings.parameters()) + list(decoder.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    PAD_ID_TOKEN = tokenizer.pad_token_id

    indexers = {'input_tokens': tokenizer, 'word_positions': word_pos_indexer, 'ku_positions': ku_pos_indexer, 'modes': mode_indexer}
    train_items = get_aligned_dataset(train_args, train_preds, train_label, train_prop, train_word_pos, train_ku_pos, train_modes, indexers)
    dev_items = get_aligned_dataset(dev_args, dev_preds, dev_label, dev_prop, dev_word_pos, dev_ku_pos, dev_modes, indexers)
    test_items = get_aligned_dataset(test_args, test_preds, test_label, test_prop, test_word_pos, test_ku_pos, test_modes, indexers)
    
    train_dataset = NTCDataset(train_items)
    dev_dataset = NTCDataset(dev_items)
    test_dataset = NTCDataset(test_items)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=INFERENCE_BATCH_SIZE, 
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=INFERENCE_BATCH_SIZE, 
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    # dev_items = get_aligned_dataset(dev_args, dev_preds, dev_label, dev_prop, dev_word_pos, dev_ku_pos, dev_modes, indexers)
    # dev_dataset = NTCDataset(dev_items)
    # train_dataloader = DataLoader(
    #     dev_dataset, 
    #     batch_size=TRAINING_BATCH_SIZE, 
    #     collate_fn=collate_fn,
    #     shuffle=False,
    #     num_workers=1, 
    #     pin_memory=True
    # )

    criterion = nn.CrossEntropyLoss()

    with Path(f'{OUTPUT_PATH}/config.json').open('w') as f:
        json.dump(hyper_parameters, f)

    max_dev_all_score = 0.0
    logs = []
    for e in range(NUM_EPOCHS):
        log = {}
        training_loss = 0.0
        embeddings.train()
        decoder.train()
        for items in train_dataloader:
            input_items = {k: allocate_data_to_device(v, DEVICE) for k, v in items.items() if k not in ['all', 'properties']}
            optimizer.zero_grad()

            x = embeddings(input_items)
            x = decoder(x)
            decoded_items = decoding_for_pointer(x['logits'], input_items['labels'])

            # [[[i, t, l, p, g, w, n, gp, wp, np, gl, wl, nl] for t, l, p, g, w, n, gp, wp, np, gl, wl, nl in zip(['null'] + items['all'][i]['tokenized_tokens'] + ['pad', 'pad'], decoded_items['labels'][i], ['null'] + items['properties'][i] + ['pad', 'pad'], torch.nn.functional.one_hot(decoded_items['positional_labels']['g'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_labels']['w'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_labels']['n'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_predictions']['g'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_predictions']['w'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_predictions']['n'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), decoded_items['logits']['g'][i].tolist(), decoded_items['logits']['w'][i].tolist(), decoded_items['logits']['n'][i].tolist())] for i in range(input_items['labels'].shape[0])]
            loss = {k: criterion(decoded_items['logits'][k], decoded_items['positional_labels'][k]) for k in decoded_items['positional_labels'].keys()}
            summed_loss = torch.stack(list(loss.values())).sum()
            summed_loss.backward()
            # for_debug = [[[i, t, l, p, g, w, n, gp, wp, np, gl, wl, nl] for t, l, p, g, w, n, gp, wp, np, gl, wl, nl in zip(['null'] + items['all'][i]['tokenized_tokens'] + ['pad', 'pad'], decoded_items['labels'][i], ['null'] + items['properties'][i] + ['pad', 'pad'], torch.nn.functional.one_hot(decoded_items['positional_labels']['g'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_labels']['w'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_labels']['n'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_predictions']['g'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_predictions']['w'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), torch.nn.functional.one_hot(decoded_items['positional_predictions']['n'][i], num_classes=input_items['labels'].shape[1]+1).tolist(), decoded_items['logits']['g'][i].tolist(), decoded_items['logits']['w'][i].tolist(), decoded_items['logits']['n'][i].tolist())] for i in range(input_items['labels'].shape[0])]

            predicted_labels = decoding_for_inference(decoded_items['positional_predictions'], items['labels'].shape[1])
            predicted_labels, gold_labels = [l[1:]for l in predicted_labels], [l[1:] for l in decoded_items['labels']]
            tp, fp, fn = get_pr_numbers2(predicted_labels, gold_labels, items['properties'])

            if GRADIENT_CLIP != -1:
                torch.nn.utils.clip_grad_norm_(embeddings.parameters(), GRADIENT_CLIP)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRADIENT_CLIP)

            optimizer.step()
            training_loss += summed_loss.item()
        log['epoch'] = e
        log['training_loss'] = training_loss

        tp_history = []
        fp_history = []
        fn_history = []
        with torch.inference_mode():
            embeddings.eval()
            decoder.eval()

            total_loss = 0.0
            for items in dev_dataloader:
                input_items = {k: allocate_data_to_device(v, DEVICE) for k, v in items.items() if k not in ['all', 'properties']}

                x = embeddings(input_items)
                x = decoder(x)
                decoded_items = decoding_for_pointer(x['logits'], input_items['labels'])

                loss = {k: criterion(decoded_items['logits'][k], decoded_items['positional_labels'][k]) for k in decoded_items['positional_labels'].keys()}
                total_loss += torch.stack(list(loss.values())).sum().item()

                predicted_labels = decoding_for_inference(decoded_items['positional_predictions'], items['labels'].shape[1])
                predicted_labels, gold_labels = [l[1:]for l in predicted_labels], [l[1:] for l in decoded_items['labels']]
                tp, fp, fn = get_pr_numbers2(predicted_labels, gold_labels, items['properties'])

                tp_history.append(tp)
                fp_history.append(fp)
                fn_history.append(fn)

            num_tp = np.sum(tp_history, axis=0)
            num_fp = np.sum(fp_history, axis=0)
            num_fn = np.sum(fn_history, axis=0)
            all_score, dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)

            precisions = []
            recalls = []
            f1s = []
            num_tn = np.array([0] * len(num_tp))
            for _tp, _fp, _fn, _tn in zip(num_tp, num_fp, num_fn, num_tn):
                precision = 0.0
                if _tp + _fp != 0:
                    precision = _tp / (_tp + _fp)
                precisions.append(precision)

                recall = 0.0
                if _tp + _fn != 0:
                    recall = _tp / (_tp + _fn)
                recalls.append(recall)

                f1 = 0.0
                if precision + recall != 0:
                    f1 = 2 * precision * recall / (precision + recall)
                f1s.append(f1)
            log['dev_all_f1'] = all_score
            log['dev_all_dep_f1'] = dep_score
            log['dev_all_zero_f1'] = zero_score
            log['dev_f1s'] = f1s
            log['dev_precisions'] = precisions
            log['dev_recalls'] = recalls
            log['dev_tps'] = num_tp.tolist()
            log['dev_fps'] = num_fp.tolist()
            log['dev_fns'] = num_fn.tolist()
            log['dev_total_loss'] = total_loss

            total_loss = 0.0
            for items in test_dataloader:
                input_items = {k: allocate_data_to_device(v, DEVICE) for k, v in items.items() if k not in ['all', 'properties']}

                x = embeddings(input_items)
                x = decoder(x)
                decoded_items = decoding_for_pointer(x['logits'], input_items['labels'])

                loss = {k: criterion(decoded_items['logits'][k], decoded_items['positional_labels'][k]) for k in decoded_items['positional_labels'].keys()}
                total_loss += torch.stack(list(loss.values())).sum().item()

                predicted_labels = decoding_for_inference(decoded_items['positional_predictions'], items['labels'].shape[1])
                predicted_labels, gold_labels = [l[1:]for l in predicted_labels], [l[1:] for l in decoded_items['labels']]
                tp, fp, fn = get_pr_numbers2(predicted_labels, gold_labels, items['properties'])

                tp_history.append(tp)
                fp_history.append(fp)
                fn_history.append(fn)

            num_tp = np.sum(tp_history, axis=0)
            num_fp = np.sum(fp_history, axis=0)
            num_fn = np.sum(fn_history, axis=0)
            all_score, dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)

            precisions = []
            recalls = []
            f1s = []
            num_tn = np.array([0] * len(num_tp))
            for _tp, _fp, _fn, _tn in zip(num_tp, num_fp, num_fn, num_tn):
                precision = 0.0
                if _tp + _fp != 0:
                    precision = _tp / (_tp + _fp)
                precisions.append(precision)

                recall = 0.0
                if _tp + _fn != 0:
                    recall = _tp / (_tp + _fn)
                recalls.append(recall)

                f1 = 0.0
                if precision + recall != 0:
                    f1 = 2 * precision * recall / (precision + recall)
                f1s.append(f1)
            log['test_all_f1'] = all_score
            log['test_all_dep_f1'] = dep_score
            log['test_all_zero_f1'] = zero_score
            log['test_f1s'] = f1s
            log['test_precisions'] = precisions
            log['test_recalls'] = recalls
            log['test_tps'] = num_tp.tolist()
            log['test_fps'] = num_fp.tolist()
            log['test_fns'] = num_fn.tolist()
            log['test_total_loss'] = total_loss

        if max_dev_all_score <= log['dev_all_f1']:
            max_dev_all_score = log['dev_all_f1']
            models = {'embeddings': embeddings.to('cpu').state_dict(), 'decoder': decoder.to('cpu').state_dict(), 'epoch': log['epoch'], 'dev_all_score': log['dev_all_f1'], 'test_all_score': log['test_all_f1'], 'log': log}
            torch.save(models, f"{OUTPUT_PATH}/models.{log['epoch']}.pth")
            embeddings = allocate_data_to_device(embeddings, DEVICE)
            decoder = allocate_data_to_device(decoder, DEVICE)



        print(f"{log['epoch']}, {log['training_loss']}, {log['dev_total_loss']}, {log['dev_all_f1']}, {log['dev_all_dep_f1']}, {log['dev_all_zero_f1']}, {log['test_total_loss']}, {log['test_all_f1']}, {log['test_all_dep_f1']}, {log['test_all_zero_f1']}")
        logs.append(log)

        with Path(f'{OUTPUT_PATH}/traininglog.json').open('a') as f:
            json.dump(log, f)
            f.write('\n')


# 単語単位の情報を積み上げて文単位の情報にする


def testcase1():
    batched_pred_positions = []
    pred_token_positions = [11, 16]
    offsets = [(0, 0), (0, 3), (3, 4), (4, 7), (7, 10), (10, 11), (11, 13), (13, 15), (15, 17), (17, 19), (19, 20), (0, 0)]
    pred_positions = []
    for i, offset in enumerate(offsets):
        if offset[0] <= pred_token_positions[0]:
            if offset[1] <= pred_token_positions[0]:
                pred_positions.append(0)
            elif pred_token_positions[0] < offset[1] <= pred_token_positions[1]:
                pred_positions.append(1)
            elif offset[1] > pred_token_positions[1]:
                pred_positions.append(1)
        elif pred_token_positions[0] < offset[0] <= pred_token_positions[1]:
            if offset[1] <= pred_token_positions[0]:
                pred_positions.append(0)
            elif pred_token_positions[0] < offset[1] <= pred_token_positions[1]:
                pred_positions.append(1)
            elif offset[1] > pred_token_positions[1]:
                pred_positions.append(1)
        elif offset[0] > pred_token_positions[1]:
            if offset[1] <= pred_token_positions[0]:
                pred_positions.append(0)
            elif pred_token_positions[0] < offset[1] <= pred_token_positions[1]:
                pred_positions.append(0)
            elif offset[1] > pred_token_positions[1]:
                pred_positions.append(0)
        else:
            pred_positions.append(-1)

    if 1 not in set(pred_positions):
        print(f'pred not found: {i}')
    batched_pred_positions.append(pred_positions.copy())


if __name__ == "__main__":
    run_modes = [
        'rinna-gpt2',
        'tohoku-bert',
        't5-base',
        'rinna-roberta',
        'nlp-waseda-roberta-base-japanese',
        'nlp-waseda-roberta-large-japanese',
        'xlm-roberta-large',
        'xlm-roberta-base',
        'rinna-japanese-gpt-1b'
    ]
    for run_mode in run_modes:
        train(run_mode=run_mode)
    # train(batch_size=2,
    #         learning_rate=0.2, optim="sgd", dropout_ratio=0.4,
    #         norm_type={'clip': 2, 'weight_decay': 0.0})
