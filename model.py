from transformers import BigBirdModel,BigBirdTokenizer,BertModel,BertTokenizer,BertTokenizerFast
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import datetime
from torchcrf import CRF


class SSR1(nn.Module):
    def __init__(self,args):
        super(SSR1, self).__init__()
        print('SSR1')
        self.encoder = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.encoder_word_embedding_fn = lambda t: self.encoder.embeddings.word_embeddings(t)

        self.re_encoder = self.encoder
        self.re_encoder_word_embedding_fn = lambda t: self.re_encoder.embeddings.word_embeddings(t)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.hidden_dim = args.hidden_dim
        self.num_tags = args.num_tags
        self.x_2_prob_z = nn.Linear(self.hidden_dim, self.num_tags)
        self.device = args.device
        self.alpha_rationle = args.alpha_rationle
        self.class_num = args.class_num
        self.classifier = nn.Linear(self.hidden_dim, self.class_num)
    
    def forward(self, input_ids,attention_mask,tag_ids=None,shortcut_id=None):
        eps = 1e-8
        output_s = self.encoder(input_ids,attention_mask)
        selector_out = output_s[0]
        batch_size = input_ids.size(0)
        feats = self.x_2_prob_z(selector_out)
        
        ###  for bert cross
        if self.training:
            if tag_ids is not None:
                output,rationale_mask = self.bert_cross(input_ids,attention_mask,feats,tag_ids,shortcut_id)
            else:
                output,rationale_mask = self.bert_rnp(input_ids, attention_mask,feats)
            return output,rationale_mask
        else:
            output_bert_cross, rationale_mask_bert_cross = self.bert_cross(input_ids,attention_mask,feats,tag_ids,shortcut_id)
            output_bert_rnp, rationale_mask_bert_rnp = self.bert_rnp(input_ids, attention_mask,feats)
            return output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp

    def bert_cross(self,input_ids,attention_mask,feats,tag_ids,shortcut_id):
        eps = 1e-8
        output_p = self.re_encoder(input_ids,attention_mask)
        pred_out = output_p[1]

        output = self.classifier(pred_out) 
        if tag_ids is None:
            mask_length = attention_mask.sum(-1).cpu().numpy()
            rationales = feats.cpu().argmax(dim=-1).numpy()
            rationale_mask = []
            for k_index,rationale in enumerate(rationales):
                rationale_mask.append(rationale[0:mask_length[k_index]])
            return output,rationale_mask
        else:  ## for training
            assert shortcut_id != None
            loss_fct = nn.CrossEntropyLoss()
            ## rationale loss
            log_likelihood = loss_fct(feats.view(-1,self.num_tags), tag_ids.view(-1))
            ## shortcut loss
            # output_shortcut = self.re_encoder(input_ids*shortcut_id.int(),attention_mask=shortcut_id)
            output_shortcut = self.re_encoder(input_ids,attention_mask=shortcut_id)
            shortcut_out = output_shortcut[1]
            shortcut_out = self.classifier(shortcut_out) 
            z_prob = F.softmax(shortcut_out,-1)  # 0.9 , 0.1
            if self.class_num == 2:
                self.shortcut_loss = (z_prob * torch.log(z_prob / torch.FloatTensor([0.5,0.5]).unsqueeze(0).unsqueeze(1).to(input_ids.device))+eps).sum(2).sum(1).mean() 
            if self.class_num == 3:
                self.shortcut_loss = (z_prob * torch.log(z_prob / torch.FloatTensor([0.33,0.33,0.33]).unsqueeze(0).unsqueeze(1).to(input_ids.device))+eps).sum(2).sum(1).mean() 
            return output,log_likelihood


    def bert_rnp(self,input_ids, attention_mask,feats):
        eps = 1e-8
        rationale_mask = []
        special_mask = torch.zeros(attention_mask.size())
        special_mask = special_mask.to(attention_mask.device)
        special_mask[:,0] = 1.0
        if self.training:
            sampled_seq = F.gumbel_softmax(feats,hard=False,dim=2)
            sampled_seq = sampled_seq[:,:,-1].unsqueeze(2)
            sampled_seq = sampled_seq * attention_mask.unsqueeze(2)
            sampled_seq = sampled_seq.squeeze(-1)
        else:
            mask_length = attention_mask.sum(-1).cpu().numpy()
            rationales = feats.cpu().argmax(dim=-1).numpy()
            rationale_mask = []
            for k_index,rationale in enumerate(rationales):
                rationale_mask.append(rationale[0:mask_length[k_index]])
            sampled_seq = torch.tensor(rationales)
            # print(sampled_seq)
            sampled_seq = sampled_seq.to(attention_mask.device)
            sampled_seq = sampled_seq * attention_mask


        sampled_seq = 1 - (1 - sampled_seq) * (1 - special_mask)

        predictor_inputs_embeds = self.re_encoder_word_embedding_fn(input_ids)

        mask_embedding = self.re_encoder_word_embedding_fn(torch.scalar_tensor(self.tokenizer.mask_token_id,dtype=torch.long,device=sampled_seq.device))
        
        masked_inputs_embeds = predictor_inputs_embeds * sampled_seq.unsqueeze(2) + mask_embedding * (1 - sampled_seq.unsqueeze(2))
        output_p = self.re_encoder(inputs_embeds = masked_inputs_embeds,attention_mask=attention_mask)

        pred_out = output_p[1]

        output = self.classifier(pred_out) 

        infor_loss = (sampled_seq.sum(-1) / (attention_mask.sum(1)+eps) ) - self.alpha_rationle
        self.infor_loss = torch.abs(infor_loss).mean()
        regular =  torch.abs(sampled_seq[:,1:] - sampled_seq[:,:-1]).sum(1) / (attention_mask.sum(1)-1+eps)
        self.regular = regular.mean()
        return output , rationale_mask



class SSR2(nn.Module):
    def __init__(self,args):
        super(SSR2, self).__init__()
        print('SSR2')
        self.encoder = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.encoder_word_embedding_fn = lambda t: self.encoder.embeddings.word_embeddings(t)

        self.re_encoder = self.encoder
        self.re_encoder_word_embedding_fn = lambda t: self.re_encoder.embeddings.word_embeddings(t)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.hidden_dim = args.hidden_dim
        self.num_tags = args.num_tags
        self.x_2_prob_z = nn.Linear(self.hidden_dim, self.num_tags)
        self.device = args.device
        self.alpha_rationle = args.alpha_rationle
        self.class_num = args.class_num
        self.classifier = nn.Linear(self.hidden_dim, self.class_num)
        self.vencoder = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.mse = MSE_div(args)
    def forward(self, input_ids,attention_mask,tag_ids=None,shortcut_id=None):
        eps = 1e-8
        output_s = self.encoder(input_ids,attention_mask)
        selector_out = output_s[0]
        batch_size = input_ids.size(0)
        feats = self.x_2_prob_z(selector_out)
        
        ###  for bert cross
        if self.training:
            if tag_ids is not None:
                output,rationale_mask = self.bert_cross(input_ids,attention_mask,feats,tag_ids,shortcut_id)
            else:
                output,rationale_mask = self.bert_rnp(input_ids, attention_mask,feats)
            return output,rationale_mask
        else:
            output_bert_cross, rationale_mask_bert_cross = self.bert_cross(input_ids,attention_mask,feats,tag_ids,shortcut_id)
            output_bert_rnp, rationale_mask_bert_rnp = self.bert_rnp(input_ids, attention_mask,feats)
            return output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp

    def bert_cross(self,input_ids,attention_mask,feats,tag_ids,shortcut_id):
        eps = 1e-8
        output_p = self.re_encoder(input_ids,attention_mask)
        pred_out = output_p[1]

        output = self.classifier(pred_out) 
        if tag_ids is None:
            mask_length = attention_mask.sum(-1).cpu().numpy()
            rationales = feats.cpu().argmax(dim=-1).numpy()
            rationale_mask = []
            for k_index,rationale in enumerate(rationales):
                rationale_mask.append(rationale[0:mask_length[k_index]])
            return output,rationale_mask
        else:  ## for training
            assert shortcut_id != None
            loss_fct = nn.CrossEntropyLoss()
            ## rationale loss
            log_likelihood = loss_fct(feats.view(-1,self.num_tags), tag_ids.view(-1))
            ## shortcut loss
            # output_shortcut = self.re_encoder(input_ids*shortcut_id.int(),attention_mask=shortcut_id)
            output_shortcut = self.vencoder(input_ids,attention_mask=shortcut_id)
            shortcut_outs = output_shortcut[1]
            self.shortcut_out = self.classifier(shortcut_outs) 
            voutput = self.vencoder(input_ids,attention_mask)[1]
            self.shortcut_loss = self.mse(shortcut_outs,voutput)
            # shortcut_out = self.classifier(shortcut_out) 
            # z_prob = F.softmax(shortcut_out,-1)  # 0.9 , 0.1
            # if self.class_num == 2:
            #     self.shortcut_loss = (z_prob * torch.log(z_prob / torch.FloatTensor([0.5,0.5]).unsqueeze(0).unsqueeze(1).to(input_ids.device))+eps).sum(2).sum(1).mean() 
            # if self.class_num == 3:
            #     self.shortcut_loss = (z_prob * torch.log(z_prob / torch.FloatTensor([0.33,0.33,0.33]).unsqueeze(0).unsqueeze(1).to(input_ids.device))+eps).sum(2).sum(1).mean() 
            return output,log_likelihood


    def bert_rnp(self,input_ids, attention_mask,feats):
        eps = 1e-8
        rationale_mask = []
        special_mask = torch.zeros(attention_mask.size())
        special_mask = special_mask.to(attention_mask.device)
        special_mask[:,0] = 1.0
        if self.training:
            sampled_seq = F.gumbel_softmax(feats,hard=False,dim=2)
            sampled_seq = sampled_seq[:,:,-1].unsqueeze(2)
            sampled_seq = sampled_seq * attention_mask.unsqueeze(2)
            sampled_seq = sampled_seq.squeeze(-1)
        else:
            mask_length = attention_mask.sum(-1).cpu().numpy()
            rationales = feats.cpu().argmax(dim=-1).numpy()
            rationale_mask = []
            for k_index,rationale in enumerate(rationales):
                rationale_mask.append(rationale[0:mask_length[k_index]])
            sampled_seq = torch.tensor(rationales)
            # print(sampled_seq)
            sampled_seq = sampled_seq.to(attention_mask.device)
            sampled_seq = sampled_seq * attention_mask


        sampled_seq = 1 - (1 - sampled_seq) * (1 - special_mask)

        predictor_inputs_embeds = self.re_encoder_word_embedding_fn(input_ids)

        mask_embedding = self.re_encoder_word_embedding_fn(torch.scalar_tensor(self.tokenizer.mask_token_id,dtype=torch.long,device=sampled_seq.device))
        
        masked_inputs_embeds = predictor_inputs_embeds * sampled_seq.unsqueeze(2) + mask_embedding * (1 - sampled_seq.unsqueeze(2))
        output_p = self.re_encoder(inputs_embeds = masked_inputs_embeds,attention_mask=attention_mask)

        pred_out = output_p[1]
        with torch.no_grad():
            voutput = self.vencoder(input_ids,attention_mask)[1]
    
        output = self.classifier(pred_out) 

        inv_output = self.classifier(voutput) 
        z_prob = F.softmax(inv_output,-1)  # 0.9 , 0.1
        if self.class_num == 2:
            self.comp_loss = (z_prob * torch.log(z_prob / torch.FloatTensor([0.5,0.5]).unsqueeze(0).unsqueeze(1).to(input_ids.device))+eps).sum(2).sum(1).mean() 
        if self.class_num == 3:
            self.comp_loss = (z_prob * torch.log(z_prob / torch.FloatTensor([0.33,0.33,0.33]).unsqueeze(0).unsqueeze(1).to(input_ids.device))+eps).sum(2).sum(1).mean() 
            
        infor_loss = (sampled_seq.sum(-1) / (attention_mask.sum(1)+eps) ) - self.alpha_rationle
        self.infor_loss = torch.abs(infor_loss).mean()
        regular =  torch.abs(sampled_seq[:,1:] - sampled_seq[:,:-1]).sum(1) / (attention_mask.sum(1)-1+eps)
        self.regular = regular.mean()
        return output , rationale_mask

