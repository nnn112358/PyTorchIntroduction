""" この例のコードは NLP プロジェクトで語彙頻度の集計と
単語表（ボキャブラリ）構築に利用できます。
"""

from collections import Counter

class Vocab(object):

    UNK = '<unk>'

    def __init__(self, counter, max_size=None, min_freq=1,
                 specials=['<unk>', '<pad>'], specials_first=True):

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        # 整数インデックスから単語への写像を定義
        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            max_size = None if max_size is None else max_size + len(specials)

        # 入力に特殊トークンがあれば除去
        for tok in specials:
            del counter[tok]

        # まずアルファベット順でソートし、その後頻度でソート
        words_and_frequencies = sorted(counter.items(), \
                                       key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # 低頻度語を除外
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if Vocab.UNK in specials: 
            unk_index = specials.index(Vocab.UNK)
            self.unk_index = unk_index if specials_first \
                else len(self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # 単語から整数インデックスへの写像を定義
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

def build_vocab_from_iterator(iterator):

    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    word_vocab = Vocab(counter)
    return word_vocab
