#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Regex to detect any CJK Unified Ideographs
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Common Chinese -> Japanese replacements.
# Order matters: put longer/more specific phrases first.
REPLACEMENTS = [
    ("《深入浅出PyTorch——从模型到源码》书籍源代码",
     "『わかりやすい PyTorch —— モデルからソースコードまで』書籍のソースコード"),
    ("如果对于代码有疑惑或者代码中有错误，请在GitHub仓库开新的Issue指出。",
     "コードに関する疑問や誤りがあれば、GitHub リポジトリで新しい Issue を作成してお知らせください。"),
    ("源代码目录", "ソースコード目次"),
    ("深度学习", "深層学習"),
    ("简介", "概要"),
    ("无代码，略", "コードなし（割愛）"),
    ("PyTorch 深度学习框架简介", "PyTorch 深層学習フレームワークの概要"),
    ("软件包", "パッケージ"),
    ("导入和测试", "インポートとテスト"),
    ("安装PyTorch的依赖关系", "PyTorch の依存関係のインストール"),
    ("编译命令", "ビルドコマンド"),
    ("Python列表和Numpy数组转换为PyTorch张量", "Python リストと NumPy 配列を PyTorch テンソルに変換"),
    ("指定形状生成张量", "指定形状のテンソルを生成"),
    ("在不同设备上的张量", "異なるデバイス上のテンソル"),
    ("张量形状相关的一些函数", "テンソル形状に関する関数"),
    ("张量的切片和索引", "テンソルのスライスとインデックス"),
    ("张量的函数运算", "テンソルの関数演算"),
    ("极值和排序", "極値とソート"),
    ("张量的矩阵乘法运算", "テンソルの行列積演算"),
    ("torch.einsum函数的使用", "torch.einsum 関数の使用"),
    ("张量的拼接和分割", "テンソルの結合と分割"),
    ("张量维度扩增和压缩", "テンソル次元の拡張と圧縮"),
    ("PyTorch模块类的构建方法", "PyTorch モジュールクラスの構築方法"),
    ("PyTorch线性回归模型示例", "PyTorch 線形回帰モデルの例"),
    ("PyTorch线性回归模型调用方法实例", "PyTorch 線形回帰モデルの呼び出し例"),
    ("PyTorch模块方法调用实例", "PyTorch モジュールメソッドの呼び出し例"),
    ("反向传播函数测试代码", "逆伝播関数のテストコード"),
    ("梯度函数的使用方法", "勾配関数の使用方法"),
    ("控制计算图产生的方法示例", "計算グラフの生成を制御する方法の例"),
    ("损失函数模块的使用方法", "損失関数モジュールの使用方法"),
    ("简单线性回归函数和优化器", "簡単な線形回帰関数とオプティマイザ"),
    ("PyTorch优化器对不同参数指定不同的学习率", "PyTorch オプティマイザでパラメータごとに異なる学習率を設定"),
    ("PyTorch学习率衰减类示例", "PyTorch 学習率減衰クラスの例"),
    ("torch.utils.data.DataLoader类的签名", "torch.utils.data.DataLoader クラスのシグネチャ"),
    ("torch.untils.data.Dataset类的构造方法", "torch.utils.data.Dataset クラスのコンストラクタ"),
    ("简单torch.utils.data.Dataset类的实现", "簡単な torch.utils.data.Dataset クラスの実装"),
    ("torch.utils.data.IterableDataset类的构造方法", "torch.utils.data.IterableDataset クラスのコンストラクタ"),
    ("PyTorch保存和载入模型", "PyTorch モデルの保存と読み込み"),
    ("PyTorch的状态字典的保存和载入", "PyTorch の state_dict の保存と読み込み"),
    ("PyTorch检查点的结构", "PyTorch チェックポイントの構造"),
    ("TensorBoard使用方法示例", "TensorBoard の使用例"),
    ("SummaryWriter提供的添加数据显示的方法", "SummaryWriter のログ追加メソッド"),
    ("torch.nn.DataParallel使用方式", "torch.nn.DataParallel の使い方"),
    ("PyTorch分布式进程启动函数", "PyTorch 分散プロセス起動関数"),
    ("多进程训练模型的数据载入", "マルチプロセス学習でのデータ読み込み"),
    ("分布式数据并行模型的API", "分散データ並列モデルの API"),
    ("分布式数据并行模型训练时的输出", "分散データ並列モデルの学習時の出力"),
    # Common boilerplate/docstrings
    ("为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，\n    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码\n    （#号及其后面内容为注释，可以忽略）",
     "以下のコードの実行結果を再現するには、PyTorch をインストールした後、\n    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。\n    （# 以降はコメントのため無視して構いません）"),
    ("该代码仅为演示函数签名和所用方法，并不能实际运行",
     "このコードは関数のシグネチャと使用方法を示すためのサンプルで、実行は想定していません"),
    ("以下代码仅作为情感分析模型的实现参考",
     "以下のコードは感情分析モデル実装の参考例です"),
    ("以下代码仅作为循环神经网络语言模型的实现参考",
     "以下のコードは RNN 言語モデル実装の参考例です"),
    ("以下代码仅作为word2vec的CBOW模型的实现参考",
     "以下のコードは word2vec の CBOW モデル実装の参考例です"),
    ("本代码仅作为Seq2Seq模型的实现参考",
     "本コードは Seq2Seq モデル実装の参考例です"),
    ("本代码仅作为BERT模型的实现参考",
     "本コードは BERT モデル実装の参考例です"),
]

CH_NUMS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10
}

def chinese_ordinal_to_arabic(text: str) -> str:
    # Convert patterns like 第3章 -> 第3章
    def repl(m):
        s = m.group(1)
        # handle 1..99 for simple forms used here
        if len(s) == 1:
            n = CH_NUMS.get(s, None)
            if n:
                return f"第{n}章"
        if s.startswith("十"):
            n = 10 + CH_NUMS.get(s[1:], 0) if len(s) > 1 else 10
            return f"第{n}章"
        if s.endswith("十"):
            n = CH_NUMS.get(s[0], 0) * 10
            return f"第{n}章"
        if len(s) == 2:
            n = CH_NUMS.get(s[0], 0) * 10 + CH_NUMS.get(s[1], 0)
            if n:
                return f"第{n}章"
        return f"第{s}章"

    return re.sub(r"第([一二三四五六七八九十]+)章", repl, text)


def apply_replacements(text: str) -> str:
    out = text
    # Convert ordinal chapter headings first
    out = chinese_ordinal_to_arabic(out)
    # Replace common tokens
    for src, dst in REPLACEMENTS:
        out = out.replace(src, dst)
    # Minor term mappings for README (Chinese -> Japanese)
    out = out.replace("目录", "目次").replace("源代码", "ソースコード")
    out = out.replace("安装", "インストール")
    out = out.replace("编译", "ビルド")
    out = out.replace("例子", "例")
    out = out.replace("示例", "例")
    return out


def process_file(path: Path) -> bool:
    try:
        s = path.read_text(encoding="utf-8")
    except Exception:
        return False
    if not CJK_RE.search(s):
        return False
    t = apply_replacements(s)
    if t != s:
        path.write_text(t, encoding="utf-8")
        return True
    return False


def main():
    exts = {".md", ".MD", ".markdown", ".py"}
    changed = 0
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        # Skip this translator script itself
        try:
            if p.resolve() == SELF:
                continue
        except Exception:
            pass
        if p.suffix in exts:
            if process_file(p):
                changed += 1
    print(f"Translated files: {changed}")


if __name__ == "__main__":
    main()
