import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
from sklearn import metrics
import numpy as np
import torch
import os
import torch.nn as nn
from utils import *
from model import SSR1

from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import prettytable as pt
from sklearn.metrics import accuracy_score, auc, average_precision_score, classification_report, precision_recall_curve, roc_auc_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch.utils import data
from short_easy_utils import *

parser = argparse.ArgumentParser(description='VMask classificer')

parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=300, help='max_len')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--hidden_dim', type=int, default=768, help='number of hidden dimension')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dev_data_dir', type=str)
parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--class_num', type=int, default=2, help='class_num')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--use_crf', type=int, default=0)
parser.add_argument('--loss_type', type=str, default='mse')
parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Linear warmup over warmup_steps.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="gradient_accumulation_steps")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, 
                        help="Epsilon for Adam optimizer.")
parser.add_argument('--fraction_rationales', type=float, default=1.0,
                        help='what fraction of sentences have rationales')
parser.add_argument('--weight_decay_finetune', type=float, default=1e-5,
                        help='weight decay finetune') 
parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")
parser.add_argument("--dataset", default="movie_reviews", type=str,
                        help="dataset")
parser.add_argument("--date", default="1009", type=str)
parser.add_argument("--fixed", default="yes", type=str)
parser.add_argument('--num_tags', type=int, default=2)
parser.add_argument('--is_da', type=str, default='no')

args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)
# logger_combine_AT

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

if args.dataset == 'movie_reviews':
    dataset_processor = MovieReviewsProcessor()
    args.alpha_rationle = 0.1
elif args.dataset == 'evinf':
    dataset_processor = EvinfProcessor()
    args.class_num = 3
    args.alpha_rationle = 0.1
elif args.dataset == 'multi_rc':            # modified add
    dataset_processor = MultiRCProcessor() # modified add
elif args.dataset == 'boolq':
    dataset_processor = BoolQProcessor() # modified delete

dataset_processor.set_fraction_rationales(args.fraction_rationales)

train_examples = dataset_processor.get_train_examples(args.data_dir)
dev_examples = dataset_processor.get_dev_examples(args.data_dir)
test_examples = dataset_processor.get_test_examples(args.data_dir)
train_shortcut_examples = dataset_processor.get_train_shortcut_examples(args.data_dir)
train_DA_examples = dataset_processor.get_train_DA_examples(args.data_dir)
train_Sem_DA_examples = dataset_processor.get_train_Sem_DA_examples(args.data_dir)


print(len(train_examples))

tag_map = dataset_processor.get_tag_map()
num_labels = dataset_processor.get_num_labels()
num_tags = dataset_processor.get_num_tags()
args.num_tags = num_tags

### for bert rnp
train_dataset = DatasetWitheasyRationales(train_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)
dev_dataset = DatasetWitheasyRationales(dev_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)
test_dataset = DatasetWitheasyRationales(test_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)
### for bert cross
train_shortcut_dataset = DatasetWithDARationales(train_shortcut_examples, tokenizer, tag_map, args.max_len, args.dataset)
### DA
train_DA_dataset = DatasetWithDARationales(train_DA_examples, tokenizer, tag_map, args.max_len, args.dataset)
train_Sem_DA_dataset = DatasetWithDARationales(train_Sem_DA_examples, tokenizer, tag_map, args.max_len, args.dataset)


train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=DatasetWitheasyRationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)

dev_dataloader = data.DataLoader(
    dataset=dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn= DatasetWitheasyRationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    #num_workers=8,
    #pin_memory=True,
    collate_fn= DatasetWitheasyRationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)

train_shortcut_dataloader = data.DataLoader(
    dataset=train_shortcut_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    #num_workers=8,
    #pin_memory=True,
    # pin_memory=True,
    collate_fn=DatasetWithDARationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)

train_DA_dataloader = data.DataLoader(
    dataset=train_DA_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    #num_workers=8,
    #pin_memory=True,
    # pin_memory=True,
    collate_fn=DatasetWithDARationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)

train_Sem_DA_dataloader = data.DataLoader(
    dataset=train_Sem_DA_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    #num_workers=8,
    #pin_memory=True,
    # pin_memory=True,
    collate_fn=DatasetWithDARationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)

for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))

def main():
    print('main_without_da')
    # load model
    # max_length = 50，rationale占比：array([0.0952381 , 0.16666667, 0.28571429])

    model = eval(args.model_name)(args)
    model = model.to(args.device)
    # for p in model.encoder.parameters():
    #     p.requires_grad = False
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)   
    # optimizer = AdamW(model.parameters(), lr=args.lr)
    
    named_params = list(model.named_parameters()) 
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params \
            if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_finetune}
    ]

    
    # total_steps = len(data_all) * args.epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.epochs+1):
        model.train()
        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
        #  for bert cross small dataset
        
        for step,batch in enumerate(tqdm(train_shortcut_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood  + 0.1*model.shortcut_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        #  for bert rnp big dataset
        
        for step,batch in enumerate(tqdm(train_dataloader)):
        
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch

            optimizer.zero_grad()
            
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            
            output,_ = model(input_ids,attention_mask)
            loss = criterion(output,label_ids) + args.alpha*model.infor_loss + args.beta*model.regular
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        if epoch>0:
            model.eval()
            ##############################  dev  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []


            for step,batch in enumerate(tqdm(dev_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)

            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')



            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' , '      F1     ' , '      percents     ' ]) 
            table.add_row(['dev-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['dev-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0], class_macro_f12, np.array(all_tag_preds_rnp).sum()/len(all_tag_gold_rnp) ])



            logger.info(table)

            # table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
            # table.add_row(['dev-cross',  class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 ])
            # table.add_row(['dev-rnp',  class_micro_f12, class_macro_precision2, class_macro_recall2, class_macro_f12 ])
            # logger.info(table)

            ##############################  test  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []


            for step,batch in enumerate(tqdm(test_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)

            test_eval = {}
            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')


            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' ,   '      F1     ' , '      percents     ' ]) 
            table.add_row(['test-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['test-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0],  class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])


            logger.info(table)

        # if epoch>10:
        #     PATH = args.save_path+args.model_name.lower()+'_'+args.is_da+'_'+str(args.date)+args.dataset+'_'+str(epoch)
        #     torch.save(model.state_dict(), PATH)



def main_with_da():
    print('main_with_da')
    model = eval(args.model_name)(args)
    model = model.to(args.device)
    # named_params = list(model.named_parameters()) 
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in named_params \
    #         if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_finetune}
    # ]
    # num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.epochs+1):
        model.train()
        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
    
        for step,batch in enumerate(tqdm(train_shortcut_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood + 0.1*model.shortcut_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        #  for DA dataset
        for step,batch in enumerate(tqdm(train_DA_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 
                
        #  for bert rnp big dataset
    
        for step,batch in enumerate(tqdm(train_dataloader)):
        
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch

            optimizer.zero_grad()
            
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            
            output,_ = model(input_ids,attention_mask)
            loss = criterion(output,label_ids) + args.alpha*model.infor_loss + args.beta*model.regular
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        if epoch>0:
            model.eval()
            ##############################  dev  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []

            token_f1_list = []
            for step,batch in enumerate(tqdm(dev_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)

                batch_size = input_ids.size(0)
               
                tag_ids = tag_ids.cpu().numpy()

                for index,rationale_ in enumerate(tag_ids):
                    bin_attrs_ = rationale_mask_bert_cross[index]
                    token_f1 = f1_score(
                        y_true=rationale_[0:len(bin_attrs_)],
                        y_pred=bin_attrs_,
                        average='macro',
                    )
                    token_f1_list.append(token_f1)

            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')



            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' , '      F1     ' , '      percents     ' ]) 
            table.add_row(['dev-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['dev-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0], class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])



            logger.info(table)

            # table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
            # table.add_row(['dev-cross',  class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 ])
            # table.add_row(['dev-rnp',  class_micro_f12, class_macro_precision2, class_macro_recall2, class_macro_f12 ])
            # logger.info(table)

            ##############################  test  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []

            token_f1_list = []
            for step,batch in enumerate(tqdm(test_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)
                batch_size = input_ids.size(0)
                tag_ids = tag_ids.cpu().numpy()
                for index,rationale_ in enumerate(tag_ids):
                    bin_attrs_ = rationale_mask_bert_cross[index]
                    token_f1 = f1_score(
                        y_true=rationale_[0:len(bin_attrs_)],
                        y_pred=bin_attrs_,
                        average='macro',
                    )
                    token_f1_list.append(token_f1)

            test_eval = {}
            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')


            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' ,   '      F1     ' , '      percents     ' ]) 
            table.add_row(['test-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['test-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0],  class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])


            logger.info(table)

        # save the last model
        # if epoch>10:
        #     PATH = args.save_path+args.model_name.lower()+'_'+args.is_da+'_'+str(args.date)+args.dataset+'_'+str(epoch)
        #     torch.save(model.state_dict(), PATH)



def main_with_sem():
    print('main_with_sem')
    model = eval(args.model_name)(args)
    model = model.to(args.device)
    # named_params = list(model.named_parameters()) 
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in named_params \
    #         if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_finetune}
    # ]
    # num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.epochs+1):
        model.train()
        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
    
        for step,batch in enumerate(tqdm(train_shortcut_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood + 0.1*model.shortcut_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        #  for DA dataset
        for step,batch in enumerate(tqdm(train_Sem_DA_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood + 0.1*model.shortcut_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 
                
        #  for bert rnp big dataset
    
        for step,batch in enumerate(tqdm(train_dataloader)):
        
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch

            optimizer.zero_grad()
            
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            
            output,_ = model(input_ids,attention_mask)
            loss = criterion(output,label_ids) + args.alpha*model.infor_loss + args.beta*model.regular
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        if epoch>0:
            model.eval()
            ##############################  dev  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []

            token_f1_list = []
            for step,batch in enumerate(tqdm(dev_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)

                batch_size = input_ids.size(0)
               
                tag_ids = tag_ids.cpu().numpy()

                for index,rationale_ in enumerate(tag_ids):
                    bin_attrs_ = rationale_mask_bert_cross[index]
                    token_f1 = f1_score(
                        y_true=rationale_[0:len(bin_attrs_)],
                        y_pred=bin_attrs_,
                        average='macro',
                    )
                    token_f1_list.append(token_f1)

            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')



            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' , '      F1     ' , '      percents     ' ]) 
            table.add_row(['dev-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['dev-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0], class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])



            logger.info(table)

            # table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
            # table.add_row(['dev-cross',  class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 ])
            # table.add_row(['dev-rnp',  class_micro_f12, class_macro_precision2, class_macro_recall2, class_macro_f12 ])
            # logger.info(table)

            ##############################  test  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []

            token_f1_list = []
            for step,batch in enumerate(tqdm(test_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)
                batch_size = input_ids.size(0)
                tag_ids = tag_ids.cpu().numpy()
                for index,rationale_ in enumerate(tag_ids):
                    bin_attrs_ = rationale_mask_bert_cross[index]
                    token_f1 = f1_score(
                        y_true=rationale_[0:len(bin_attrs_)],
                        y_pred=bin_attrs_,
                        average='macro',
                    )
                    token_f1_list.append(token_f1)

            test_eval = {}
            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')


            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' ,   '      F1     ' , '      percents     ' ]) 
            table.add_row(['test-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['test-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0],  class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])


            logger.info(table)

        # save the last model
        # if epoch>10:
        #     PATH = args.save_path+args.model_name.lower()+'_'+args.is_da+'_'+str(args.date)+args.dataset+'_'+str(epoch)
        #     torch.save(model.state_dict(), PATH)


def main_with_mixed():
    print('main_with_mixed')
    model = eval(args.model_name)(args)
    model = model.to(args.device)
    # named_params = list(model.named_parameters()) 
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in named_params \
    #         if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_finetune}
    # ]
    # num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.epochs+1):
        model.train()
        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
    
        for step,batch in enumerate(tqdm(train_shortcut_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood + 0.1*model.shortcut_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        for step,batch in enumerate(tqdm(train_DA_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            
            #### BoostRNP baseline
            
            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 
        #  for DA dataset
        for step,batch in enumerate(tqdm(train_Sem_DA_dataloader)):
            # if step>2:break
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels,shortcut_ids = batch
            optimizer.zero_grad()
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            shortcut_ids = shortcut_ids.to(args.device)
            

            output,log_likelihood= model(input_ids,attention_mask,tag_ids,shortcut_ids)
            loss = criterion(output,label_ids) + log_likelihood + 0.1*model.shortcut_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 
                
        #  for bert rnp big dataset
    
        for step,batch in enumerate(tqdm(train_dataloader)):
        
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch

            optimizer.zero_grad()
            
            input_ids = input_ids.to(args.device)
            tag_ids  =  tag_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label_ids = label_ids.to(args.device)
            
            output,_ = model(input_ids,attention_mask)
            loss = criterion(output,label_ids) + args.alpha*model.infor_loss + args.beta*model.regular
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() 

        if epoch>0:
            model.eval()
            ##############################  dev  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []

            token_f1_list = []
            for step,batch in enumerate(tqdm(dev_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)

                batch_size = input_ids.size(0)
               
                tag_ids = tag_ids.cpu().numpy()

                for index,rationale_ in enumerate(tag_ids):
                    bin_attrs_ = rationale_mask_bert_cross[index]
                    token_f1 = f1_score(
                        y_true=rationale_[0:len(bin_attrs_)],
                        y_pred=bin_attrs_,
                        average='macro',
                    )
                    token_f1_list.append(token_f1)

            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')



            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' , '      F1     ' , '      percents     ' ]) 
            table.add_row(['dev-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['dev-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0], class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])



            logger.info(table)

            # table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
            # table.add_row(['dev-cross',  class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 ])
            # table.add_row(['dev-rnp',  class_micro_f12, class_macro_precision2, class_macro_recall2, class_macro_f12 ])
            # logger.info(table)

            ##############################  test  ##############################
            percents = 0
            predictions_charge_cross = []
            true_charge_cross = []

            predictions_charge_rnp = []

            all_tag_preds_cross = []
            all_tag_gold_cross = []

            all_tag_preds_rnp = []
            all_tag_gold_rnp = []

            token_f1_list = []
            for step,batch in enumerate(tqdm(test_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                true_charge_cross.extend(label_ids.cpu().numpy())
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                tag_ids = tag_ids.to(args.device)
                label_ids = label_ids.to(args.device)
                with torch.no_grad():
                    output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask)

                pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                predictions_charge_cross.extend(pred_cross)

                pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                predictions_charge_rnp.extend(pred_rnp)

                pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_cross.extend(pred_tags_flat)
                all_tag_gold_cross.extend(gold_tags_flat)
                
                assert len(gold_tags_flat) == len(pred_tags_flat)
                pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                all_tag_preds_rnp.extend(pred_tags_flat)
                all_tag_gold_rnp.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat)
                batch_size = input_ids.size(0)
                tag_ids = tag_ids.cpu().numpy()
                for index,rationale_ in enumerate(tag_ids):
                    bin_attrs_ = rationale_mask_bert_cross[index]
                    token_f1 = f1_score(
                        y_true=rationale_[0:len(bin_attrs_)],
                        y_pred=bin_attrs_,
                        average='macro',
                    )
                    token_f1_list.append(token_f1)

            test_eval = {}
            precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
            print(true_charge_cross)
            print(predictions_charge_cross)
            print(predictions_charge_rnp)
            precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

            class_macro_precision, class_macro_recall, class_macro_f1, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
            class_macro_precision2, class_macro_recall2, class_macro_f12, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')


            table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' ,   '      F1     ' , '      percents     ' ]) 
            table.add_row(['test-cross',precision_tagging[0], recall_tagging[0], f1_tagging[0],class_macro_f1,  np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross)])

            table.add_row(['test-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0],  class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])


            logger.info(table)

        # save the last model
        # if epoch>10:
        #     PATH = args.save_path+args.model_name.lower()+'_'+args.is_da+'_'+str(args.date)+args.dataset+'_'+str(epoch)
        #     torch.save(model.state_dict(), PATH)



if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    if args.is_da == 'no':
        main()
    if args.is_da == 'yes':
        main_with_da()
    if args.is_da == 'sem':
        main_with_sem()
    if args.is_da == 'mixed':
        main_with_mixed()
