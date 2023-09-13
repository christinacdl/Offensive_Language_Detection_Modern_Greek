# -*- coding: utf-8 -*-


#!pip install torchvision seaborn torchtext wordsegment transformers contractions torch lightning emoji unidecode ekphrasis -U SentencePiece focal_loss_torch pytorch_metric_learning flair

# Import necessary libraries
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")
import torch
for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")
device = torch.device("cuda:0")

from torchtext import data, datasets
import random
import re
import glob
import emoji
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,  matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, RobertaForSequenceClassification, DebertaForSequenceClassification,XLMRobertaXLConfig, XLMRobertaXLForSequenceClassification, AutoModelForSequenceClassification, AutoConfig, AutoModel, get_linear_schedule_with_warmup, BertTokenizer, DebertaV2ForSequenceClassification, BertModel,BertConfig,DebertaV2Config, DebertaV2Model,  AlbertConfig, AlbertModel, AlbertTokenizer, RobertaModel, RobertaTokenizer, RobertaConfig, DebertaConfig, DebertaModel, DebertaTokenizer, AutoTokenizer, DebertaV2Tokenizer
import matplotlib.pyplot as plt
import wordsegment as ws
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

"""**THE LABELS**"""

# Determine the number of labels in the data and map each of these labels to an index. There are 2 labels: Not Offensive (NOT) and Offensive (OFF)
labels_to_id = {'NOT': 0 ,'OFFENSIVE': 1}
id_to_labels = {0 : 'NOT', 1: 'OFFENSIVE'}

"""**FUNCTIONS USED**"""

# A function that sets seed for reproducibility
def set_seed(seed_value):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)

# A function that checks if a directory exists else creates the directory
def check_create_path(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print('Directory created at {}'.format(path))
  else:
    print('Directory {} already exists!'.format(path))

# A function that reads a csv or tsv file
def read_a_file(filename):
  # Check first whether a certain file or directory exists
  if os.path.exists(filename):
    print('Current file opened: ',[os.path.join(filename, file) for file in glob.glob(filename)])

    # Find the file extension to open it properly
    find_separator = {'.csv': ',', '.tsv': '\t'}
    basename, format = os.path.splitext(filename)
    assert format in find_separator
    separator = find_separator[format]

    # Read different extensions of files using pandas with 2 different separators
    read_file = pd.read_csv(filename, sep = separator, encoding = 'utf-8')

    return read_file

  else:
    print('File or directory not accessible. Please check the filename and ensure that the entered path of the file is in "tsv" or "csv" form.')


# A function that opens and reads a dataset either from Hugging Face or from a local directory
def open_dataset(dataset_path, text_column, label_column, huggingface_dataset = True, type_split = 1, labelled_dataset = True):

  if huggingface_dataset:
    if type_split == 1:
      dataset = load_dataset(dataset_path, split='train')
      read_file = pd.DataFrame(dataset)
    elif type_split == 2:
      dataset = load_dataset(dataset_path, split='validation')
      read_file = pd.DataFrame(dataset)
    elif type_split == 3:
      dataset = load_dataset(dataset_path, split='test')
      read_file = pd.DataFrame(dataset)
    else:
      print('Please specify the number "1" for train set, "2" for validation and "3" to use the test set.')

  else:
    read_file = read_a_file(dataset_path)

  read_file.info()

  # Get the keys and their corresponding number of values
  keys = read_file.keys()
  for key in keys:
    df_len = len(read_file[key].unique()) # the length of the unique values of each column
    print('{0:25}{1:10}'.format(key,df_len))

  # Remove missing values and keep the dataFrame with valid entries in the same variable
  read_file.dropna(inplace = True)

  # Remove the index
  read_file.reset_index(inplace = True, drop = True)

  if text_column != 'text':
    read_file = read_file.rename({text_column:'text'}, axis = 1)
  else:
    read_file

  if label_column != 'label':
    read_file = read_file.rename({label_column:'label'}, axis = 1)
  else:
    read_file

  print(read_file.label.value_counts())

  if labelled_dataset:
    # Encode the concatenated data
    encoded_texts = [tokenizer.encode(sent, add_special_tokens = True) for sent in read_file.text.values]

    # Find the maximum length
    max_len = max([len(sent) for sent in encoded_texts])
    print('Maximum sentence length: ', max_len)

    # Find the minimum length
    min_len = min([len(sent) for sent in encoded_texts])
    print('Minimum sentence length: ', min_len)

  else:
    None

  return read_file

def replaceMultiple(main, replacements, new):
  for elem in replacements:
    if elem in main:
      main = main.replace(elem, new)
  return main

def normalize(x):
  x = x.replace('ά', 'α')
  x = x.replace('έ', 'ε')
  x = x.replace('ή', 'η')
  x = replaceMultiple(x, ['ί', 'ΐ', 'ϊ'], 'ι')
  x = x.replace('ό', 'ο')
  x = replaceMultiple(x, ['ύ', 'ΰ', 'ϋ'], 'υ')
  x = x.replace('ώ', 'ω')
  return x

ws.load()
def segment_hashtags(tweet):
  tweet = re.sub('#[\S]+', lambda match: ' '.join(ws.segment(match.group())), tweet)
  return tweet

def clean_text(x):
  puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
              '*', '+', '\\', '•', '~', '@', '£',
              '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
              '█', '½', 'à', '…',
              '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
              '¥', '▓', '—', '‹', '─',
              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
              '¾', 'Ã', '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
              '¹', '≤', '‡', '√', ]

  x = str(x)
  for punct in puncts:
    x = x.replace(punct, f' {punct} ')
  return x

def sep_digits(x):
  return " ".join(re.split('(\d+)', x))

def sep_punc(x):
  punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~؛،؟؛.»«”'
  out = []
  for char in x:
    if char in punc:
      out.append(' ' + char + ' ')
    else:
      out.append(char)
  return ''.join(out)


def preprocessing_greek(text):
  # Replace the RT with whitespace
  text = re.sub('RT @\w+: ','', text)
  # Remove the @user tags
  text = re.sub(r'@[A-Za-z0-9_]+','', text)
  text = re.sub(r'username', '', text)
  # Remove the url links
  text = re.sub(r'http\S+', '', text)
  # Remove the 'url' and 'html' word
  url_words = ['url', 'URL', 'html', 'HTML', 'http', 'HTTP']
  for u in url_words:
    text = re.sub(u,'', text)
  # Convert the emojis into their textual representation
  text = emojis_into_text(text)
  # Replace '&amp;' with 'και'
  text = re.sub(r'&amp;','και', text)
  text = re.sub(r'&','και', text)
  # Replace the unicode apostrophe
  text = re.sub(r"’","'", text)
  text = normalize(text)
  text = sep_digits(text)
  text = sep_punc(text)
  text = str(text).lower()
  text = re.sub(r'rt',' ', text)
  # Remove the extra whitespace
  text = re.sub(' +',' ', text)
  return text

def emojis_into_text(sentence):
  demojized_sent = emoji.demojize(sentence)
  emoji_txt = re.sub(r':[\S]+:', lambda x: x.group().replace('_', ' ').replace('-', ' ').replace(':', ''), demojized_sent)
  return emoji_txt

# A function that splits the data into training and validation
def data_splitting(dataframe, text_column, label_column, split_ratio):
  x_train_texts, y_val_texts, x_train_labels, y_val_labels = train_test_split(dataframe[text_column], dataframe[label_column],
                                                                                random_state = 42,
                                                                                test_size = split_ratio,
                                                                                stratify = dataframe[label_column])
  print(f'Dataset split into train and validation/test sets using {split_ratio} split ratio.')
  train_df = pd.concat([x_train_texts, x_train_labels], axis = 1)
  val_df = pd.concat([y_val_texts, y_val_labels], axis = 1)
  print(f'Size of training set: {len(train_df)}')
  print(f'Size of validation/test set: {len(val_df)}')
  return train_df, val_df

#=== METRICS CALCULATION ===

def compute_metrics(p):
  logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  preds = np.argmax(logits, axis=1)
  macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
  micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
  accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
  return {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'accuracy': accuracy}

def show_confusion_matrix(true, predicted, class_names, threshold):
  cm = confusion_matrix(true, predicted)
  df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
  hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Labels')
  plt.xlabel('Predicted Labels')
  plt.savefig(os.path.join(args['results_data_directory'], 'f{threshold}_confusion_matrix'), bbox_inches='tight')

# A function that calculates all the metrics using the validation/test set
def calculate_metrics(y_true, preds, threshold):
  print('\nCALCULATING METRICS...')
  assert len(preds) == len(y_true)

  # Calculate the accuracy of the model
  acc = accuracy_score(y_true, preds)

  # Calculate the Matthews Correlation Coefficient
  # -1 indicates total disagreement between predicted classes and actual classes
  # 0 is synonymous with completely random guessing
  # 1 indicates total agreement between predicted classes and actual classes
  mcc = matthews_corrcoef(y_true, preds)

  # Calculate model's metrics
  model_f1_score = f1_score(y_true, preds, average = 'macro', zero_division = 1)
  model_precision = precision_score(y_true, preds, average = 'macro', zero_division = 1)
  model_recall = recall_score(y_true, preds, average = 'macro', zero_division = 1)

  # Calculate general precision, recall, F1 score of each class
  precision, recall, fscore, support = score(y_true, preds, zero_division = 1)
  print(f'Accuracy: {acc}')
  print(f'F1 score: {model_f1_score}')
  print(f'Precision: {model_precision}')
  print(f'Recall : {model_recall}')
  print(f'Matthews Correlation Coefficient: {mcc}')
  print(f'\nPrecision of each class: {precision}')
  print(f'Recall of each class: {recall}')
  print(f'F1 score of each class: {fscore}')

  # Print the classification report
  class_names = ['NOT', 'OFFENSIVE']
  print(classification_report(y_true, preds, target_names = class_names))

  # Create the confusion matrix
  cm = confusion_matrix(labels, preds)
  df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
  hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Labels')
  plt.xlabel('Predicted Labels')
  plt.savefig(args['results_data_directory'] + f'{threshold}_confusion_matrix.png', bbox_inches='tight')
  plt.show()
  plt.close()

  return model_f1_score, model_precision, model_recall, mcc, acc, precision, recall, fscore, support 


def tokenize(batch):
  return tokenizer(batch['text'], max_length = args['max_seq_length'], padding='max_length', truncation=True)
                   

"""**NECESSARY & IMPORTANT PARAMETERS**"""

dict_BERT_model_names = {1: 'bert-base-uncased',
                         2: 'bert-large-uncased',
                         3: 'bert-base-multilingual-uncased', #multilingual YES
                         4: 'bert-base-multilingual-cased',
                         5: 'dimitriz/greek-media-bert-base-uncased', #multilingual YES threshold_run_6
                         6: 'nlpaueb/bert-base-greek-uncased-v1'}   #multilingual YES threshold_run_9

dict_AlBERT_model_names = {1: 'albert-base-v1',
                           2: 'albert-base-v2',
                           3: 'albert-xlarge-v1',
                           4: 'albert-xlarge-v2',
                           5: 'albert-xxlarge-v1',
                           6: 'albert-xxlarge-v2'}

dict_RoBERTa_model_names = {1: 'roberta-base',
                            2: 'roberta-large'}

dict_DeBERTa_model_names = {1: 'microsoft/deberta-base',
                            2: 'microsoft/deberta-large',
                            3: 'microsoft/deberta-xlarge',
                            4: 'microsoft/deberta-v2-xlarge',
                            5: 'microsoft/deberta-v2-xxlarge',
                            6: 'microsoft/deberta-v3-large',
                            7: 'microsoft/mdeberta-v3-base'} #multilingual

dict_XLM_RoBERTa_model_names = {1: 'xlm-roberta-base',   #multilingual
                                2: 'xlm-roberta-large'}  #multilingual

dict_multilingual_model_names = {1: 'studio-ousia/mluke-base', #multilingual
                                 2: 'cvcio/comments-el-toxic',
                                 3: 'autopilot-ai/EthicalEye'} #multilingual 


args = {'task_name': 'Offensive Language Detection in Greek Corpus',
        'data_directory': '/home/geoten/Projects/christodoulou/train_test_files/',
        'new_data_directory':  '/home/geoten/Projects/christodoulou/train_test_files/prepared_files/threshold_run_10/',            
        'results_data_directory': '/home/geoten/Projects/christodoulou/predictions/threshold_run_10/',             
        'output_model_directory': '/home/geoten/Projects/christodoulou/GREEK_OFFENSIVE_DETECTION/',
        'model_type':  'XLM_RoBERTa',   # Write the model's name 
        'model_name': str(dict_XLM_RoBERTa_model_names[1]),   # Change the index to train with the model of your choice
        'cache_dir': 'cache/', 
        'num_classes': 2,   # Binary classification 
        'max_seq_length': 512,
        'data_split_ratio': 0.2,
        'train_batch_size': 16,
        'validation_batch_size': 16,
        'num_train_epochs': 5, 
        'warmup_steps': 0, 
        'max_grad_norm' : 1.0,
        'weight_decay':  0.01,   
        'learning_rate': 2e-5,   
        'adam_epsilon': 1e-8,    
        'gradient_accumulation_steps': 1,
        'early_stopping_patience': 4, 
        'seed': 42}

print('================',str(args['task_name']),'================\n')

# Get the directory names and the specific model used
print('Output directory: ' + str(args['output_model_directory']))
print('Model Name: ' + str(args['model_name']))
args['output_specific_model_dir'] = args['output_model_directory'] + args['model_name'] + '/' + 'threshold_run_10'
print('Output Directory: ' + str(args['output_specific_model_dir']))

# Check whether the directories exist else create them
print('\nChecking that the necessary paths exist...')
check_create_path(args['data_directory'])
check_create_path(args['new_data_directory'])
check_create_path(args['output_model_directory'])
check_create_path(args['results_data_directory'])
check_create_path(args['output_specific_model_dir'])

MODEL_CLASSES = {'BERT': (BertConfig, BertForSequenceClassification, BertTokenizer),
                 'AlBERT': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
                 'RoBERTa': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
                 'DeBERTa': (DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer),
                 'DeBERTaV2': (DebertaV2Config, DebertaV2ForSequenceClassification, DebertaV2Tokenizer),
                 'XLM_RoBERTa': (XLMRobertaXLConfig, XLMRobertaXLForSequenceClassification, AutoTokenizer),
                 'other': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

"""**INITIALIZE THE PRETRAINED MODEL AND ITS TOKENIZER**"""

# Set seed for reproducibility
set_seed(args['seed'])

config = config_class.from_pretrained(args['model_name'],
                                      num_labels = args['num_classes'],
                                      finetuning_task = args['task_name'])

tokenizer = tokenizer_class.from_pretrained(args['model_name'],
                                            do_lower_case = False)

model = model_class.from_pretrained(args['model_name'],
                                                   num_labels = args['num_classes'], # The number of output labels
                                                   output_attentions = False,        # Whether the model returns attentions weights
                                                   output_hidden_states = False,
                                                   id2label = id_to_labels,
                                                   label2id = labels_to_id,
                                                   ignore_mismatched_sizes=True)

def convert_scores(score, threshold):

  if score >= threshold:
    label = 1
  else:
    label = 0
  return label

f1_scores = []
precision_scores = []
recall_scores = []
mcc_scores = []
accuracy_scores = []
threshold_v = []
flagged_re = []
precisions = []
recalls = []
f1s = []
support_classes = []

column_names=['id', 'text', 'score']

AIKIA = pd.read_csv(args['data_directory'] + 'updated_corpus_gold_corrected.tsv', sep ='\t', header = None, names = column_names,  skipinitialspace = True)
print(AIKIA.info())

AIKIA['text'] = AIKIA['text'].apply(lambda x: preprocessing_greek(x))


AIKIA['text'] = AIKIA['text'].replace('“','"', regex=True)
AIKIA['text'] = AIKIA['text'].replace('"',"'", regex=True) 
AIKIA['text'] = AIKIA['text'].str.rstrip()

print('Number of duplicates in the file:', AIKIA.duplicated(subset ='text').sum())
AIKIA.drop_duplicates(subset ='text', keep = 'first', inplace = True)
print('Number of duplicates in the file:', AIKIA.duplicated(subset ='text').sum())


thresholds = np.arange(0.2, 0.8, step = 0.05) 

for threshold_value in thresholds:
  
  AIKIA[f'{threshold_value}'] = AIKIA['score'].apply(convert_scores, threshold = threshold_value)

  print(f'SPLITTING INTO TRAIN, VALIDATION AND TEST SETS USING {threshold_value} threshold value...')
  training_set1, test_set = data_splitting(AIKIA, 'text', f'{threshold_value}', 0.2) 
  training_set, val_set = data_splitting(training_set1, 'text', f'{threshold_value}', 0.1) 
  
  training_set = training_set.rename(columns = {f'{threshold_value}':'label'})
  val_set = val_set.rename(columns = {f'{threshold_value}':'label'})
  test_set = test_set.rename(columns = {f'{threshold_value}':'label'})

  print(training_set['label'].value_counts())
  print(test_set['label'].value_counts())
  print(val_set['label'].value_counts())

  training_set.to_json(args['new_data_directory'] + f'{threshold_value}_train.json', force_ascii=False, orient='records', lines=True)
  val_set.to_json(args['new_data_directory'] + f'{threshold_value}_val.json', force_ascii=False, orient='records', lines=True)
  test_set.to_json(args['new_data_directory'] + f'{threshold_value}_test.json', force_ascii=False, orient='records', lines=True)

  # Get value counts for each set
  train_value_counts = training_set['label'].value_counts() 
  test_value_counts = test_set['label'].value_counts() 
  val_value_counts = val_set['label'].value_counts()

  # Get the count of a specific class
  class_count_train_0 = train_value_counts[0]
  class_count_train_1 = train_value_counts[1]
  class_count_test_0 = test_value_counts[0]
  class_count_test_1 = test_value_counts[1]
  class_count_val_0 = val_value_counts[0]
  class_count_val_1 = val_value_counts[1]
    
  value_counts_df = pd.DataFrame({'train_counts':[class_count_train_0, class_count_train_1],
                                  'test_counts': [class_count_test_0, class_count_test_1],
                                  'val_counts':[class_count_val_0, class_count_val_1]})

  # Save value counts to file
  value_counts_df.to_csv(args['new_data_directory'] + f'{threshold_value}_value_counts.csv', encoding = 'utf-8', index = False, header = True, sep =',')


  print(f'USING TRAIN, VALIDATION AND TEST DATASETS WITH THRESHOLD {threshold_value}...\n')
  train_dataset = load_dataset('json', data_files= args['new_data_directory'] + f'{threshold_value}_train.json', split = 'train')
  val_dataset = load_dataset('json', data_files= args['new_data_directory'] + f'{threshold_value}_val.json', split = 'train')
  test_dataset = load_dataset('json', data_files= args['new_data_directory'] + f'{threshold_value}_test.json', split = 'train')

  print('\nTOKENIZING DATASETS...')
  train_dataloader = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
  val_dataloader = val_dataset.map(tokenize, batched=True, remove_columns=['text'])
  test_dataloader = test_dataset.map(tokenize, batched=True, remove_columns=['text'])

  train_dataloader.set_format(type='torch',columns=['input_ids', 'attention_mask', 'label'])
  val_dataloader.set_format(type='torch',columns=['input_ids', 'attention_mask', 'label'])
  test_dataloader.set_format(type='torch',columns=['input_ids', 'attention_mask', 'label'])
  print(train_dataloader.features.keys())
  print(val_dataloader.features.keys())
  print(test_dataloader.features.keys())
  print()

  repository_id = args['output_specific_model_dir'] + f'AIKIA_{threshold_value}'

  # Initialize training arguments 
  arguments = TrainingArguments(
    output_dir = repository_id,
    logging_dir = f'{repository_id}/logs',
    evaluation_strategy= 'epoch',
    save_strategy= 'epoch',
    eval_steps = 500,
    save_total_limit = 2,
    learning_rate = args['learning_rate'],
    num_train_epochs = args['num_train_epochs'],
    metric_for_best_model = 'macro_f1',
    greater_is_better = True,
    weight_decay = args['weight_decay'],
    load_best_model_at_end = True,
    per_device_train_batch_size = args['train_batch_size'],
    per_device_eval_batch_size = args['validation_batch_size'],
    overwrite_output_dir = True,
    fp16 = False,
    fp16_full_eval = False,
    seed = args['seed'],
    warmup_steps = args['warmup_steps'],
    gradient_accumulation_steps = args['gradient_accumulation_steps'],
    optim='adamw_torch_fused',
    logging_strategy = 'steps',
    logging_steps = 500,
    push_to_hub = False,
    hub_strategy = 'every_save')

  # Initialize the Trainer 
  trainer = Trainer(
    model = model,
    args = arguments,
    train_dataset = train_dataloader,
    eval_dataset = val_dataloader,
    compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = args['early_stopping_patience'])])

  # Train model
  train_result = trainer.train()

  tokenizer.save_pretrained(repository_id)

  # Train metrics and model save
  metrics = train_result.metrics
  metrics['train_samples'] = len(train_dataloader)
  trainer.save_model()
  trainer.log_metrics('train', metrics)
  trainer.save_metrics('train', metrics)
  trainer.save_state()

  # Eval metrics
  metrics = trainer.evaluate(eval_dataset = val_dataloader)
  max_eval_samples = len(val_dataloader)
  metrics['eval_samples'] = max_eval_samples
  trainer.log_metrics('eval', metrics)
  trainer.save_metrics('eval', metrics)

  # Predict metrics and Classification Report test_dataloader
  predictions, labels, metrics = trainer.predict(test_dataloader, metric_key_prefix='predict')
  max_predict_samples = len(test_dataloader)
  metrics['predict_samples'] = len(test_dataloader)

  # Log metrics
  trainer.log_metrics('predict', metrics)
  trainer.save_metrics('predict', metrics)

  preds = np.argmax(predictions, axis=-1)

  # Calculate performance metrics on test set
  f1, precision, recall, mcc, accuracy, precision_class, recall_class, f1_class, support = calculate_metrics(labels, preds, threshold_value)

  df_true = pd.DataFrame(labels, columns = ['True'])
  df_preds = pd.DataFrame(preds, columns = ['Prediction'])

  df_metrics = pd.DataFrame([[threshold_value, f1, precision, recall, mcc, accuracy, precision_class, recall_class, f1_class, support]], 
                            columns = ['Threshold', 'F1', 'Precision', 'Recall', 'MCC', 'Accuracy', 'Precisions', 'Recalls', 'F1s', 'Support'])

  # Concatenate id, text, true labels and predicted labels
  final_true_preds = pd.concat([df_true, df_preds], axis = 1) 

  # Change the numerical labels to categorical labels
  final_true_preds['True'].replace(id_to_labels, inplace = True)
  final_true_preds['Prediction'].replace(id_to_labels, inplace = True)

  # Save the new file
  final_true_preds.to_csv(args['results_data_directory'] + f'True_Predictions_{threshold_value}.csv', encoding = 'utf-8', index = False, header = True, sep =',')
  
  # Append metrics to lists 
  f1_scores.append(f1)
  precision_scores.append(precision)
  recall_scores.append(recall)
  mcc_scores.append(mcc)
  accuracy_scores.append(accuracy)
  threshold_v.append(threshold_value)
  precisions.append(precision_class)
  recalls.append(recall_class)
  f1s.append(f1_class) 
  support_classes.append(support)

metrics_dict = {'Threshold': threshold_v, 'Macro_F1': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores, 
                'MCC': mcc_scores, 'Accuracy': accuracy_scores, 'Precisions': precisions, 'Recalls': recalls, 'F1s': f1s, 'Support': support_classes}

all_metrics_thresholds = pd.DataFrame(metrics_dict)
print(all_metrics_thresholds)
all_metrics_thresholds.to_csv(args['results_data_directory'] + f'All_Metrics.csv', encoding = 'utf-8', index = False, header = True, sep =',')
