# coding: utf-8

import dgl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Optional
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from layers import GIN, RGCN, RGIN, HGT
from aser_utils import NUM_EDGETYPE


class MultipleChoiceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = AutoConfig.from_pretrained(config['model_checkpoint'])
        self.roberta = AutoModel.from_pretrained(config['model_checkpoint'])

        # The following params are from Roberta's config
        # https://huggingface.co/roberta-base/resolve/main/config.json
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(input_ids.size(), attention_mask.size())
        # torch.Size([32, 2, 78]) torch.Size([32, 2, 78])
        outputs, num_choices = self.LM_forward(input_ids, attention_mask)

        pooled_output = outputs[1]
        # print(pooled_output.shape) # [2*5, 768]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # print(logits.shape) # [2*5, 1]
        
        reshaped_logits = logits.view(-1, num_choices)
        

        loss = self.get_loss(reshaped_logits, labels)

        if not self.config.use_return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        num_choices = input_ids.shape[1] if input_ids is not None else None

        outputs = self.roberta(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        return outputs, num_choices

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss




class GraphMCModel(torch.nn.Module):
    def __init__(self, args, num_choices=2):
        super().__init__()
        self.args = args
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(args.encoder_name)
        self.encoder = AutoModel.from_pretrained(args.encoder_name)
        self.encoder.resize_token_embeddings(args.tokenizer_len)

        lm_hidden = self.config.hidden_size
        conv_hidden = args.conv_hidden

        self.conv_layer = GIN(args.conv_layers, num_mlp_layers=2, input_dim=lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='sum')
        self.pooling = SumPooling()

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        # self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.out_batchnorm = torch.nn.BatchNorm1d(conv_hidden)
        self.classifier = torch.nn.Linear(conv_hidden, 1)

    def forward(self, graph=None, tokenized=None, node2textid=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        graph = graph.to(self.encoder.device)
        pooled_output = self.LM_forward(*tokenized)

        # reshape feature to form feature for the input graphs
        feats = pooled_output[node2textid]

        # graph conv
        outputs = self.conv_layer(graph, feats)

        # unbatch graphs
        # graph.ndata['attr'] = torch.stack(outputs).sum(0)
        graph.ndata['attr'] = outputs[-1]
        graphs = dgl.unbatch(graph)
        pooled_output = torch.cat([self.pooling(graph_i, graph_i.ndata['attr']) for graph_i in graphs])

        pooled_output = self.out_batchnorm(pooled_output)
        logits = self.classifier(pooled_output)
        # print(logits.shape) # [2*5, 1]
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        pooled_output = outputs[1]
        return pooled_output

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss

class TextGraphMCModel(torch.nn.Module):
    def __init__(self, args, num_choices=2):
        super().__init__()
        self.args = args
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(self.args.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name)
        self.encoder.resize_token_embeddings(self.args.tokenizer_len)

        if 'deberta' in self.args.encoder_name:
            from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
            self.deberta_pooler = ContextPooler(self.config)
            Dropout = StableDropout
        else:
            Dropout = torch.nn.Dropout

        lm_hidden = self.config.hidden_size
        conv_hidden = self.args.conv_hidden

        if self.args.conv_type == 'GIN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=2, input_dim=lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='sum')
        elif self.args.conv_type == 'GCN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=1, input_dim = lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='mean')
        elif self.args.conv_type == 'HGT':
            num_heads = 4
            num_ntypes = 2
            if self.args.add_rev_edges:
                num_etypes = NUM_EDGETYPE['bi']
            else:
                num_etypes = NUM_EDGETYPE['uni']
            self.conv_layer = HGT(num_layers=self.args.conv_layers, input_dim=lm_hidden, hidden_dim=conv_hidden, num_heads=num_heads, num_ntypes=num_ntypes, num_etypes=num_etypes)
        else:
            if self.args.add_rev_edges:
                num_rels = NUM_EDGETYPE['bi']
            else:
                num_rels = NUM_EDGETYPE['uni']
            
            if self.args.conv_type == 'RGCN':
                self.conv_layer = RGCN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False)
            elif self.args.conv_type == 'RGIN':
                self.conv_layer = RGIN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False, num_mlp_layers=2)


        self.pooling = SumPooling()

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        self.lm_linear = torch.nn.Sequential(
            Dropout(self.config.hidden_dropout_prob),
            torch.nn.Linear(lm_hidden, conv_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(conv_hidden)
        )

        self.conv_batchnorm = torch.nn.BatchNorm1d(conv_hidden)
        if args.fusion_type == 'add':
            classifier_input = conv_hidden
            self.fusion = lambda x1,x2: x1+x2
        elif args.fusion_type == 'concat':
            classifier_input = conv_hidden * 2
            self.fusion = lambda x1,x2: torch.cat([x1, x2], dim=-1)
        self.classifier = torch.nn.Linear(classifier_input, 1,)

    def forward(self, graph=None, tokenized=None, text_tokenized=None, node2textid=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(text_tokenized[0].shape)
        x = self.LM_forward(*text_tokenized)
        x1 = self.lm_linear(x)

        graph = graph.to(self.encoder.device)
        pooled_output = self.LM_forward(*tokenized)

        # reshape feature to form feature for the input graphs
        feats = pooled_output[node2textid]
        # graph conv
        outputs = self.conv_layer(graph, feats, kwargs.get('etypes', None))
        
        # unbatch graphs
        # graph.ndata['attr'] = torch.stack(outputs).sum(0)
        graph.ndata['attr'] = outputs[-1]
        graphs = dgl.unbatch(graph)
        pooled_output = torch.cat([self.pooling(graph_i, graph_i.ndata['attr']) for graph_i in graphs])

        x2 = self.conv_batchnorm(pooled_output)
        x = self.fusion(x1, x2)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        if 'deberta' in self.args.encoder_name:
            pooled_output = self.deberta_pooler(outputs[0])
        else:
            pooled_output = outputs[1]
        return pooled_output

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss

class Attention(torch.nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(n_hidden, n_hidden))

    def forward(self, s, node_embeds):
        ''' s: 1*n, node_embeds: j*n'''
        h = torch.mm(node_embeds, self.weight)
        h = torch.einsum('jn,n->j', h, s)
        sh = torch.nn.functional.softmax(h)
        pooled_embeds = torch.einsum('j, jn->n', sh, node_embeds)
        return h, pooled_embeds

class TextGraphMCModel_Att(torch.nn.Module):
    def __init__(self, args, num_choices=2):
        super().__init__()
        self.args = args
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(self.args.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name)
        self.encoder.resize_token_embeddings(self.args.tokenizer_len)

        if 'deberta' in self.args.encoder_name:
            from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
            self.deberta_pooler = ContextPooler(self.config)
            Dropout = StableDropout
        else:
            Dropout = torch.nn.Dropout

        lm_hidden = self.config.hidden_size
        conv_hidden = self.args.conv_hidden

        if self.args.conv_type == 'GIN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=2, input_dim=lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='sum')
        elif self.args.conv_type == 'GCN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=1, input_dim = lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='mean')
        else:
            if self.args.add_rev_edges:
                num_rels = NUM_EDGETYPE['bi']
            else:
                num_rels = NUM_EDGETYPE['uni']
            
            if self.args.conv_type == 'RGCN':
                self.conv_layer = RGCN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False)
            elif self.args.conv_type == 'RGIN':
                self.conv_layer = RGIN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False, num_mlp_layers=2)


        # self.pooling = SumPooling()
        self.pooling = Attention(conv_hidden)

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        self.lm_linear = torch.nn.Sequential(
            Dropout(self.config.hidden_dropout_prob),
            torch.nn.Linear(lm_hidden, conv_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(conv_hidden)
        )

        self.conv_batchnorm = torch.nn.BatchNorm1d(conv_hidden)
        if args.fusion_type == 'add':
            classifier_input = conv_hidden
            self.fusion = lambda x1,x2: x1+x2
        elif args.fusion_type == 'concat':
            classifier_input = conv_hidden * 2
            self.fusion = lambda x1,x2: torch.cat([x1, x2], dim=-1)
        self.classifier = torch.nn.Linear(classifier_input, 1,)

    def forward(self, graph=None, tokenized=None, text_tokenized=None, node2textid=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(text_tokenized[0].shape)
        x = self.LM_forward(*text_tokenized)
        x1 = self.lm_linear(x)

        graph = graph.to(self.encoder.device)
        pooled_output = self.LM_forward(*tokenized)

        # reshape feature to form feature for the input graphs
        feats = pooled_output[node2textid]
        # graph conv
        outputs = self.conv_layer(graph, feats, kwargs.get('etypes', None))
        
        # unbatch graphs
        # graph.ndata['attr'] = torch.stack(outputs).sum(0)
        graph.ndata['attr'] = outputs[-1]
        graphs = dgl.unbatch(graph)

        batched_pooled_graph_embeds = []
        for i, graph_i in enumerate(graphs):
            attn_weight, pooled_embeds = self.pooling(x1[i], graph_i.ndata['attr'])
            batched_pooled_graph_embeds.append(pooled_embeds.unsqueeze(0))
        pooled_output = torch.cat(batched_pooled_graph_embeds)

        x2 = self.conv_batchnorm(pooled_output)
        x = self.fusion(x1, x2)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}


    def attn_forward(self, graph=None, tokenized=None, text_tokenized=None, node2textid=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(text_tokenized[0].shape)
        x = self.LM_forward(*text_tokenized)
        x1 = self.lm_linear(x)

        graph = graph.to(self.encoder.device)
        pooled_output = self.LM_forward(*tokenized)

        # reshape feature to form feature for the input graphs
        feats = pooled_output[node2textid]
        # graph conv
        outputs = self.conv_layer(graph, feats, kwargs.get('etypes', None))
        
        # unbatch graphs
        # graph.ndata['attr'] = torch.stack(outputs).sum(0)
        graph.ndata['attr'] = outputs[-1]
        graphs = dgl.unbatch(graph)

        batched_pooled_graph_embeds = []
        attentions = []
        for i, graph_i in enumerate(graphs):
            attn_weight, pooled_embeds = self.pooling(x1[i], graph_i.ndata['attr'])
            batched_pooled_graph_embeds.append(pooled_embeds.unsqueeze(0))
            attentions.append(attn_weight.unsqueeze(0))
        pooled_output = torch.cat(batched_pooled_graph_embeds)

        x2 = self.conv_batchnorm(pooled_output)
        x = self.fusion(x1, x2)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits, 'attn': attentions}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        if 'deberta' in self.args.encoder_name:
            pooled_output = self.deberta_pooler(outputs[0])
        else:
            pooled_output = outputs[1]
        return pooled_output

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss


class TextGraphMCModel_NEEG(torch.nn.Module):
    def __init__(self, args, num_choices=2):
        super().__init__()
        self.args = args
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(self.args.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name)
        self.encoder.resize_token_embeddings(self.args.tokenizer_len)

        if 'deberta' in self.args.encoder_name:
            from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
            self.deberta_pooler = ContextPooler(self.config)
            Dropout = StableDropout
        else:
            Dropout = torch.nn.Dropout

        lm_hidden = self.config.hidden_size
        conv_hidden = self.args.conv_hidden

        if self.args.conv_type == 'GIN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=2, input_dim=lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='sum')
        elif self.args.conv_type == 'GCN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=1, input_dim = lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='mean')
        elif self.args.conv_type == 'HGT':
            num_heads = 4
            num_ntypes = 2
            if self.args.add_rev_edges:
                num_etypes = NUM_EDGETYPE['bi']
            else:
                num_etypes = NUM_EDGETYPE['uni']
            self.conv_layer = HGT(num_layers=self.args.conv_layers, input_dim=lm_hidden, hidden_dim=conv_hidden, num_heads=num_heads, num_ntypes=num_ntypes, num_etypes=num_etypes)
        else:
            if self.args.add_rev_edges:
                num_rels = NUM_EDGETYPE['bi']
            else:
                num_rels = NUM_EDGETYPE['uni']
            
            if self.args.conv_type == 'RGCN':
                self.conv_layer = RGCN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False)
            elif self.args.conv_type == 'RGIN':
                self.conv_layer = RGIN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False, num_mlp_layers=2)

        self.pooling = SumPooling()

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        self.lm_output = torch.nn.Sequential(
            Dropout(self.config.hidden_dropout_prob),
            # torch.nn.Linear(lm_hidden, conv_hidden),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(conv_hidden)
        )

        self.conv_output = torch.nn.Sequential(
            torch.nn.BatchNorm1d(conv_hidden),
            torch.nn.Linear(conv_hidden, lm_hidden),
        )
        # self.conv_batchnorm = torch.nn.BatchNorm1d(conv_hidden)
        if args.fusion_type == 'add':
            classifier_input = lm_hidden
            self.fusion = lambda x1,x2: x1+x2
        elif args.fusion_type == 'concat':
            classifier_input = lm_hidden * 2
            self.fusion = lambda x1,x2: torch.cat([x1, x2], dim=-1)
        self.classifier = torch.nn.Linear(classifier_input, 1,)

    def forward(self, graph=None, tokenized=None, text_tokenized=None, node2textid=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(text_tokenized[0].shape)
        x = self.LM_forward(*text_tokenized)
        x1 = self.lm_output(x)

        graph = graph.to(self.encoder.device)
        pooled_output = self.LM_forward(*tokenized)

        # reshape feature to form feature for the input graphs
        feats = pooled_output[node2textid]
        # graph conv
        outputs = self.conv_layer(graph, feats, kwargs.get('etypes', None))
        
        # unbatch graphs
        # graph.ndata['attr'] = torch.stack(outputs).sum(0)
        graph.ndata['attr'] = outputs[-1]
        graphs = dgl.unbatch(graph)
        pooled_output = torch.cat([self.pooling(graph_i, graph_i.ndata['attr']) for graph_i in graphs])

        # x2 = self.conv_batchnorm(pooled_output)
        x2 = self.conv_output(pooled_output)
        x = self.fusion(x1, x2)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        if 'deberta' in self.args.encoder_name:
            pooled_output = self.deberta_pooler(outputs[0])
        else:
            pooled_output = outputs[1]
        return pooled_output

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks.
    From huggingface: 
    https://github.com/huggingface/transformers/blob/e0be053e433d641e3a2965da114b618b4bfdbf1a/src/transformers/models/roberta/modeling_roberta.py#L1435
    """

    def __init__(self, config, num_labels=5, Dropout=nn.Dropout):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if getattr(config, 'classifier_dropout', None) is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TextGraphClassificationModel(torch.nn.Module):
    def __init__(self, args, num_choices=5):
        super().__init__()
        self.args = args
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(self.args.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name)
        self.encoder.resize_token_embeddings(self.args.tokenizer_len)

        if 'deberta' in self.args.encoder_name:
            from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
            self.deberta_pooler = ContextPooler(self.config)
            Dropout = StableDropout
        else:
            Dropout = torch.nn.Dropout

        lm_hidden = self.config.hidden_size
        conv_hidden = self.args.conv_hidden

        if self.args.conv_type == 'GIN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=2, input_dim=lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='sum')
        elif self.args.conv_type == 'GCN':
            self.conv_layer = GIN(self.args.conv_layers, num_mlp_layers=1, input_dim = lm_hidden, hidden_dim=conv_hidden, learn_eps=False, neighbor_pooling_type='mean')
        elif self.args.conv_type == 'HGT':
            num_heads = 4
            num_ntypes = 2
            if self.args.add_rev_edges:
                num_etypes = NUM_EDGETYPE['bi']
            else:
                num_etypes = NUM_EDGETYPE['uni']
            self.conv_layer = HGT(num_layers=self.args.conv_layers, input_dim=lm_hidden, hidden_dim=conv_hidden, num_heads=num_heads, num_ntypes=num_ntypes, num_etypes=num_etypes)
        else:
            if self.args.add_rev_edges:
                num_rels = NUM_EDGETYPE['bi']
            else:
                num_rels = NUM_EDGETYPE['uni']
            
            if self.args.conv_type == 'RGCN':
                self.conv_layer = RGCN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False)
            elif self.args.conv_type == 'RGIN':
                self.conv_layer = RGIN(self.args.conv_layers, lm_hidden, conv_hidden, num_rels, regularizer='basis', num_bases=self.args.num_bases, self_loop=False, num_mlp_layers=2)


        self.pooling = SumPooling()

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        # self.lm_output = torch.nn.Sequential(
        #     torch.nn.Dropout(self.config.hidden_dropout_prob),
        #     # torch.nn.Linear(lm_hidden, conv_hidden),
        #     # torch.nn.ReLU(),
        #     # torch.nn.BatchNorm1d(conv_hidden)
        # )
        self.conv_output = torch.nn.Sequential(
            torch.nn.BatchNorm1d(conv_hidden),
            torch.nn.Linear(conv_hidden, lm_hidden),
        )
        # self.conv_batchnorm = torch.nn.BatchNorm1d(conv_hidden)
        if args.fusion_type == 'add':
            self.fusion = lambda x1,x2: x1+x2
        elif args.fusion_type == 'concat':
            self.fusion = lambda x1,x2: torch.cat([x1, x2], dim=-1)
        self.classifier = ClassificationHead(self.config, num_labels=num_choices, Dropout=Dropout)

    def forward(self, graph=None, tokenized=None, text_tokenized=None, node2textid=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(text_tokenized[0].shape)
        x1 = self.LM_forward(*text_tokenized)
        # x1 = x

        graph = graph.to(self.encoder.device)
        pooled_output = self.LM_forward(*tokenized)

        # reshape feature to form feature for the input graphs
        feats = pooled_output[node2textid]
        # graph conv
        outputs = self.conv_layer(graph, feats, kwargs.get('etypes', None))
        
        # unbatch graphs
        # graph.ndata['attr'] = torch.stack(outputs).sum(0)
        graph.ndata['attr'] = outputs[-1]
        graphs = dgl.unbatch(graph)
        pooled_output = torch.cat([self.pooling(graph_i, graph_i.ndata['attr']) for graph_i in graphs])

        x2 = self.conv_output(pooled_output)
        x = self.fusion(x1, x2)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        if 'deberta' in self.args.encoder_name:
            pooled_output = self.deberta_pooler(outputs[0])
        else:
            pooled_output = outputs[1]
        return pooled_output
        # return outputs[0][:, 0, :]  # <s>

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss

class TextMCModel(torch.nn.Module):
    def __init__(self, args, num_choices=2):
        super().__init__()
        self.args = args
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(self.args.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name)
        self.encoder.resize_token_embeddings(self.args.tokenizer_len)

        if 'deberta' in self.args.encoder_name:
            from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
            self.deberta_pooler = ContextPooler(self.config)
            Dropout = StableDropout
        else:
            Dropout = torch.nn.Dropout

        lm_hidden = self.config.hidden_size
        conv_hidden = self.args.conv_hidden

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        self.lm_linear = torch.nn.Sequential(
            Dropout(self.config.hidden_dropout_prob),
            # torch.nn.Linear(lm_hidden, conv_hidden),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(conv_hidden)
        )

        # self.conv_batchnorm = torch.nn.BatchNorm1d(conv_hidden)
        # if args.fusion_type == 'add':
        #     classifier_input = conv_hidden
        #     self.fusion = lambda x1,x2: x1+x2
        # elif args.fusion_type == 'concat':
        #     classifier_input = conv_hidden * 2
        #     self.fusion = lambda x1,x2: torch.cat([x1, x2], dim=-1)
        # self.classifier = torch.nn.Linear(classifier_input, 1,)
        self.classifier = torch.nn.Linear(lm_hidden, 1,)

    def forward(self, text_tokenized=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # print(text_tokenized[0].shape)
        x = self.LM_forward(*text_tokenized)
        x = self.lm_linear(x)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        if 'deberta' in self.args.encoder_name:
            pooled_output = self.deberta_pooler(outputs[0])
        else:
            pooled_output = outputs[1]
        return pooled_output

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss


class TextMCModel_SCT(torch.nn.Module):
    def __init__(self, args, num_choices=2):
        super().__init__()
        self.args = args
        if isinstance(args, dict):
            encoder_name = args['encoder_name'] if 'encoder_name' in args else args['model_checkpoint']
            self.encoder_name = encoder_name
            conv_hidden = args['conv_hidden'] if 'conv_hidden' in args else 128
        else:
            encoder_name = args.encoder_name
            conv_hidden = args.conv_hidden
        self.num_choices = num_choices
        
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        # self.encoder.resize_token_embeddings(self.args.tokenizer_len)

        if 'deberta' in self.encoder_name:
            from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
            self.deberta_pooler = ContextPooler(self.config)
            Dropout = StableDropout
        else:
            Dropout = torch.nn.Dropout

        lm_hidden = self.config.hidden_size

        # # The following params are from Roberta's config
        # # https://huggingface.co/roberta-base/resolve/main/config.json
        self.lm_linear = torch.nn.Sequential(
            Dropout(self.config.hidden_dropout_prob),
            torch.nn.Linear(lm_hidden, conv_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(conv_hidden)
        )

        classifier_input = conv_hidden
        self.classifier = torch.nn.Linear(classifier_input, 1,)
        # self.classifier = torch.nn.Linear(lm_hidden, 1,)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        text_tokenized = (input_ids, attention_mask)
        # print(text_tokenized[0].shape)
        x = self.LM_forward(*text_tokenized)
        x = self.lm_linear(x)

        logits = self.classifier(x)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = self.get_loss(reshaped_logits, labels)

        return {'loss': loss, 'logits': reshaped_logits}
    
    def LM_forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        """ Language Model forward. 
        """

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        if 'deberta' in self.encoder_name:
            pooled_output = self.deberta_pooler(outputs[0])
        else:
            pooled_output = outputs[1]
        return pooled_output

    def get_loss(self, reshaped_logits, labels):
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss