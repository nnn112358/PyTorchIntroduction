""" 以下のコードは RNN 言語モデル実装の参考例です
"""

import torch
import torch.nn as nn

# モデル
class LM(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False):
        super(LM, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, \
                                  nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid)

# 学習コード
model = LM(ntokens, ninp, nhid, nlayers, dropout, tie_weights).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(model, iterator, optimizer, criterion):
    model.train()
    # 語彙サイズを取得
    ntokens = len(vocab)

    for data, targets in train_data:
        # data / targets の形状は L×N（L: 系列長, N: バッチ）
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        hidden = model.init_hidden(data.size(1))
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return model

# 予測コード
def evaluate(model):
    model.eval()
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    words = []
    for i in range(max_words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().softmax(-1).cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.fill_(word_idx)
        word = vocab.itos[word_idx]
        words.append(word)
    return words
