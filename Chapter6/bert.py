""" 本コードは BERT モデル実装の参考例です
"""

import torch
import torch.nn as nn

# 埋め込み層
class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # トークン埋め込み
        self.word_embeddings = nn.Embedding(config.vocab_size, 
                                            config.hidden_size, padding_idx=0)
        # 位置埋め込み
        self.position_embeddings = nn.Embedding(\
                                        config.max_position_embeddings,
                                        config.hidden_size)
        # セグメント埋め込み
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                          config.hidden_size)
        # 層正規化
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        # ドロップアウト層
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        # 入力 `input_ids` のサイズは L×N（L: 最大系列長, N: ミニバッチ）と仮定
        seq_length = input_ids.size(0)
        if position_ids is None:
            position_ids = torch.arange(seq_length, 
                                        dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + \
                     token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# BERT エンコーダ
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([nn.TransformerEncoderLayer(
                                        config.hidden_size,
                                        config.num_attention_heads,
                                        config.intermediate_size,
                                        config.hidden_dropout_prob) \
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        # 中間層の出力を逐次計算
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         head_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # 最終出力・中間出力・自己アテンション重みを含む
        return outputs  

# BERT 事前学習モデル
class BertPretrainModel(nn.Module):
    def __init__(self, config):
        super(BertPretrainModel, self).__init__()
        self.embedding = BertEmbeddings(config)
        self.bert = BertEncoder(config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.activ2 = nn.ReLU()
        self.norm = nn.LayerNorm(config.hidden_size,
                                 eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 2)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):

        embed = self.embedding(input_ids, segment_ids)
        h = self.bert(embed, input_mask, masked_pos)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf

# BERT 機械読解タスク
class BertQA(nn.Module)
    def __init__(self, config):
        super(BertQA, self).__init__()
        self.embedding = BertEmbeddings(config)
        self.bert = BertEncoder(config)

        self.start = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.end = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.activ1 = nn.Tanh()
    
    def forward(self, input_ids, segment_ids, input_mask, masked_pos):

        embed = self.embedding(input_ids, segment_ids)
        h = self.bert(embed, input_mask, masked_pos)
        h = self.activ1(self.fc(h))

        logits_start = (h*self.start).sum(-1)
        logits_end = (h*self.end).sum(-1)

        return logits_start, logits_end
