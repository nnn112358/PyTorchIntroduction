""" 以下のコードの実行結果を再現するには、PyTorch と Scikit-Learn を
    インストールした後、システムのコマンドラインで `python` を実行して
    対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',]
X = vectorizer.fit_transform(corpus)
X.toarray()
vectorizer.vocabulary_
