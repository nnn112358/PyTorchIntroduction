""" 本コードは Seq2Seq モデル実装の参考例です
"""

import torch
import torch.nn as nn

# エンコーダ
class LSTMEncoder(nn.Module):
    def __init__(
        self, dictionary,embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        padding_value=0.):

        super(LSTMEncoder, self).__init__()
        self.dictionary = dictionary
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        # 語彙中の '<PAD>' に対応するインデックスを取得
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim,
                                      self.padding_idx)

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        # 入力は L×N（L: 最大系列長, N: ミニバッチ）
        seqlen, bsz = src_tokens.size()
        # 入力系列に対応する埋め込みを取得
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # 入力テンソルをパック
        packed_x = nn.utils.rnn.pack_padded_sequence(x,
                                                     src_lengths.data.tolist())
        # 隠れ状態のサイズを取得
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        # 隠れ状態をゼロで初期化
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        # 入力テンソルから LSTM の出力を計算
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_ x, (h0, c0))
        # LSTM の出力をアンパック
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs,
                                                padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        # 融合双向LSTM的次元
        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1)\
                        .transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask \
                if encoder_padding_mask.any() else None
        }

# アテンション機構
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim,
                 output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim,
                                 source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim,
                                 output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):

          # 入力は B×H（B: ミニバッチ, H: 隠れ状態次元）
          # source_hids は L×B×H（L: 系列長）
        x = self.input_proj(input)

          # アテンションスコアを計算
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

          # パディングトークンのスコアを -inf に設定
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)

        attn_scores = F.softmax(attn_scores, dim=0)

          # エンコーダ出力の重み付き平均
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores

# デコーダ
class LSTMDecoder(nn.Module):

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512):

        super(LSTMDecoder, self).__init__()
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.need_attn = True

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units,
                                              hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim \
                    if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        self.attention = AttentionLayer(hidden_size, encoder_output_units,
                                        hidden_size, bias=False)
        self.fc_out = Linear(out_embed_dim, num_embeddings,
                             dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, 
                incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        # 予測で使用する出力トークンを取得
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # 予測で使用する状態を取得
        cached_state = utils.get_incremental_state(self, incremental_state,
                                                      'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) \
                                for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []

        # 反復して RNN 計算を実施
        for j in range(seqlen):
            # 前ステップのアテンション情報を入力へ導入
            input = torch.cat((x[j, :, :], input_feed), dim=1)
            # すべての RNN 層を反復
            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                input = F.dropout(hidden, p=self.dropout_out,
                                  training=self.training)

                prev_hiddens[i] = hidden
                prev_cells[i] = cell
            # アテンションの出力と重みを計算
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)
            input_feed = out
            outs.append(out)
            
        # 保存隠れ状態
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        attn_scores = attn_scores.transpose(0, 2)
        x = self.fc_out(x)
        return x, attn_scores
