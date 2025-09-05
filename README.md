# 『わかりやすい PyTorch —— モデルからソースコードまで』書籍のソースコード

## ソースコード目次

## 1. 第1章 深層学習の概念概要

（コードなし、割愛）

## 2. 第2章 PyTorch 深層学習フレームワークの概要

**コード2.1** [PyTorch パッケージのインポートとテスト](./Chapter2/ex_2_1.py)

**コード2.2** [PyTorch の依存関係のインストール](./Chapter2/ex_2_2.sh)

**コード2.3** [PyTorch のビルドコマンド](./Chapter2/ex_2_3.sh)

**コード2.4** [Python リストと NumPy 配列を PyTorch テンソルに変換](./Chapter2/ex_2_4.py)

**コード2.5** [指定形状のテンソルを生成](./Chapter2/ex_2_5.py)

**コード2.6-2.7** [指定形状のテンソルを生成](./Chapter2/ex_2_6.py)

**コード2.8** [異なるデバイス上のテンソル](./Chapter2/ex_2_8.py)

**コード2.9** [テンソル形状に関する関数](./Chapter2/ex_2_9.py)

**コード2.10** [テンソルのスライスとインデックス](./Chapter2/ex_2_10.py)

**コード2.11** [テンソルの関数演算](./Chapter2/ex_2_11.py)

**コード2.12** [テンソルの四則演算](./Chapter2/ex_2_12.py)

**コード2.13**  [極値とソートの関数](./Chapter2/ex_2_13.py)

**コード2.14**  [テンソルの行列積演算](./Chapter2/ex_2_14.py)

**コード2.15**  [torch.einsum 関数の使用](./Chapter2/ex_2_15.py)

**コード2.16**  [テンソルの結合と分割](./Chapter2/ex_2_16.py)

**コード2.17**  [テンソル次元の拡張と圧縮](./Chapter2/ex_2_17.py)

**コード2.18**  [テンソルのブロードキャスト](./Chapter2/ex_2_18.py)

**コード2.19** [PyTorch モジュールクラスの構築方法](./Chapter2/ex_2_19.py)

**コード2.20** [PyTorch 線形回帰モデルの例](./Chapter2/ex_2_20.py)

**コード2.21** [PyTorch 線形回帰モデルの呼び出し例](./Chapter2/ex_2_21.py)

**コード2.22** [PyTorch モジュールメソッドの呼び出し例](./Chapter2/ex_2_22.py)

**コード2.23** [逆伝播関数のテストコード](./Chapter2/ex_2_23.py)

**コード2.24** [勾配関数の使用方法](./Chapter2/ex_2_24.py)

**コード2.25** [計算グラフの生成を制御する方法の例](./Chapter2/ex_2_25.py)

**コード2.26** [損失関数モジュールの使用方法](./Chapter2/ex_2_26.py)

**コード2.27** [简单線形回帰関数和オプティマイザ](./Chapter2/ex_2_27.py)

**コード2.28** [PyTorchオプティマイザ对不同パラメータ指定不同的学習率](./Chapter2/ex_2_28.py)

**コード2.29** [PyTorch学習率減衰クラス例](./Chapter2/ex_2_29.py)

**コード2.30** [torch.utils.data.DataLoaderクラス的シグネチャ](./Chapter2/ex_2_30.py)

**コード2.31** [torch.untils.data.Datasetクラス的コンストラクタ](./Chapter2/ex_2_31.py)

**コード2.32** [简单torch.utils.data.Datasetクラス的实现](./Chapter2/ex_2_32.py)

**コード2.33** [torch.utils.data.IterableDatasetクラス的コンストラクタ](./Chapter2/ex_2_33.py)

**コード2.34** [PyTorch保存と読み込みモデル](./Chapter2/ex_2_34.py)

**コード2.35** [PyTorch的状态辞書的保存と読み込み](./Chapter2/ex_2_35.py)

**コード2.36** [PyTorch チェックポイントの構造](./Chapter2/ex_2_36.py)

**コード2.37** [TensorBoard使用方法例](./Chapter2/ex_2_37.py)

**コード2.38** [SummaryWriter提供的添加データ显示的方法](./Chapter2/ex_2_38.py)

**コード2.39** [torch.nn.DataParallel の使い方](./Chapter2/ex_2_39.py)

**コード2.40** [PyTorch分散プロセス启动関数](./Chapter2/ex_2_40.py)

**コード2.41** [多プロセス学習モデル的データ読み込み](./Chapter2/ex_2_41.py)

**コード2.42** [分散データ並列モデル的API](./Chapter2/ex_2_42.py)

**コード2.43** [分散データ並列モデル学習时的输出](./Chapter2/ex_2_43.py)

## 3. 第3章 PyTorch コンピュータビジョンモジュール

**コード3.1-3.3** [線形層の定義と使い方](./Chapter3/ex_3_1.py)

**コード3.4** [ConvNd クラスの定義コード](./Chapter3/ex_3_4.py)

**コード3.5-3.9** [正規化モジュールの定義](./Chapter3/ex_3_5.py)

**コード3.10-3.15** [プーリングモジュールの定義](./Chapter3/ex_3_10.py)

**コード3.16** [ドロップアウト層モジュールの定義](./Chapter3/ex_3_16.py)

**コード3.17** [順序モジュールのコンストラクタ](./Chapter3/ex_3_17.py)

**コード3.18** [ModuleList/ModuleDict のコンストラクタ](./Chapter3/ex_3_18.py)

**コード3.19-3.22** [AlexNet 例と特徴抽出モジュール](./Chapter3/ex_3_19.py)

**コード3.23-3.24** [ゲイン係数の計算とパラメータ初期化](./Chapter3/ex_3_23.py)

**コード3.25** [InceptionNet の基本フレーム](./Chapter3/ex_3_25.py)

**コード3.26-3.27** [ResNet の基本フレーム](./Chapter3/ex_3_26.py)

## 4. 第4章 PyTorch コンピュータビジョン事例

**コード4.1** [PyTorch の代表的データセットラッパ](./Chapter4/ex_4_1.py)

**コード4.2-4.9** [LeNet モデル一式](./Chapter4/LeNet)

**コード4.10-4.12** [argparse で LeNet のハイパラ指定](./Chapter4/ex_4_10.py)

**コード4.13** [ImageNet 読み込み部分](./Chapter4/ex_4_13.py)

**コード4.14** [ResNet ボトルネック残差モジュール](./Chapter4/ex_4_14.py)

**コード4.15-4.20** [InceptionNet サブモジュール](./Chapter4/ex_4_15.py)

**コード4.21-4.26** [SSD モデル実装](./Chapter4/SSD)

**コード4.27-4.30** [FCN モデル実装](./Chapter4/fcn.py)

**コード4.31** [U-Net モデル実装](./Chapter4/unet.py)

**コード4.32-4.37** [画像スタイル転送](./Chapter4/style_transfer.py)

**コード4.38-4.39** [変分オートエンコーダ（VAE）](./Chapter4/vae.py)

**コード4.40-4.42** [敵対的生成ネットワーク（GAN）](./Chapter4/gan.py)

## 5. 第5章 PyTorch 自然言語処理モジュール

**コード5.1** [sklearn CountVectorizer で語頻度特徴](./Chapter5/ex_5_1.py)

**コード5.2** [CountVectorizer クラス定義](./Chapter5/ex_5_2.py)

**コード5.3** [TF-IDF の例](./Chapter5/ex_5_3.py)

**コード5.4** [TfidfTransformer / TfidfVectorizer 定義](./Chapter5/ex_5_4.py)

**コード5.5** [nn.Embedding クラス定義](./Chapter5/ex_5_5.py)

**コード5.6** [単語埋め込みモジュールの使用例](./Chapter5/ex_5_6.py)

**コード5.7** [事前学習埋め込み行列からの初期化](./Chapter5/ex_5_7.py)

**コード5.8** [nn.EmbeddingBag クラス定義](./Chapter5/ex_5_8.py)

**コード5.9-5.10** [pack_padded_sequence などの使用](./Chapter5/ex_5_9.py)

**コード5.11** [単純 RNN のパラメータ例](./Chapter5/ex_5_11.py)

**コード5.12** [RNN 使用例](./Chapter5/ex_5_12.py)

**コード5.13** [LSTM/GRU のパラメータ定義](./Chapter5/ex_5_13.py)

**コード5.14** [LSTM/GRU モジュールの使い方](./Chapter5/ex_5_14.py)

**コード5.15** [RNNCell/LSTMCell/GRUCell のパラメータ](./Chapter5/ex_5_15.py)

**コード5.16** [RNNCell/LSTMCell/GRUCell の使い方](./Chapter5/ex_5_16.py)

**コード5.17** [MultiheadAttention のパラメータ](./Chapter5/ex_5_17.py)

**コード5.18** [Transformer エンコーダ/デコーダの定義](./Chapter5/ex_5_18.py)

**コード5.19** [Transformer エンコーダ/デコーダ/モデル](./Chapter5/ex_5_19.py)

## 6. 第6章 PyTorch 自然言語処理の事例

**コード6.1** [collections.Counter で語彙表を構築](./Chapter6/ex_6_1.py)

**コード6.2-6.3** [CBOW モデルと学習過程](./Chapter6/word2vec.py)

**コード6.4** [CosineSimilarity のパラメータと使用](./Chapter6/ex_6_4.py)

**コード6.5-6.6** [感情分析向け深層学習モデル](./Chapter6/sentiment.py)

**コード6.7-6.9** [RNN ベース言語モデル](./Chapter6/lm.py)

**コード6.10-6.12** [Seq2Seqモデルコード](./Chapter6/seq2seq.py)

**コード6.13-6.16** [BERTモデルコード](./Chapter6/bert.py)

## 7. 第7章 その他の重要モデル

**コード7.1** [Wide & Deep モデル](./Chapter7/wide_deep.py)

**コード7.2** [CTC 損失関数の定義](./Chapter7/ex_7_2.py)

**コード7.3** [DeepSpeechモデルコード](./Chapter7/deep_speech.py)

**コード7.4-7.7** [Tacotronモデルコード](./Chapter7/tacotron.py)

**コード7.8-7.9** [WaveNetモデルコード](./Chapter7/wavenet.py)

**コード7.10-7.14** [DQNモデルコード](./Chapter7/dqn.py)

**コード7.15-7.17** [半精度モデルの学習](./Chapter7/half_prec.py)

## 8. 第8章 PyTorch 応用

**コード8.1-8.10** [PyTorch のカスタム活性化関数/勾配](./Chapter8/GELU)

**コード8.11-8.14** [PyTorch フックの使い方](./Chapter8/ex_8_11.py)

**コード8.15-8.17** [PyTorch 静的グラフの使い方](./Chapter8/ex_8_15.py)

**コード8.18-8.22** [PyTorch 静的モデルの保存と読み込み](./Chapter8/ex_8_18)

## 9. 第9章 PyTorch ソースコード解析 

**コード9.1** [native_functions.yaml の宣言](./ex_9_1.yaml)

**コード9.2-9.4** [pybind11 の簡単な例](./py_cpp_interface)
