"""
    poetry run python -m sandbox.analyze_metadata
"""

"""
    ダウンロードしたamazon datasetの分析用。
    geminiに書かせた処理だとテキストデータを入れるとレコード数が減ってしまいそうだが、
    実際に見てみると全てのメタデータがインタラクションに紐づいている。
"""

import pandas as pd

METADATA_PATH = './Data/amazon/metadata.json'
INTRACRIONS_PATH = './Data/amazon/amazon.json'

if __name__ == '__main__':
    print("analyze dataset...")

    df_meta = pd.read_json(METADATA_PATH , lines=True)

    df_meta.info()

    print("len of data")
    print(len(df_meta))
    print("")

    print("rate of not-null records")
    print(df_meta.notnull().mean().sort_values(ascending=False))
    print("")

    print("analyze interactions...")

    df_interactions = pd.read_json(INTRACRIONS_PATH, lines=True)
    print("num unique asins:")
    print(df_meta['asin'].nunique() / df_interactions['asin'].nunique())

    m = pd.merge(df_interactions['asin'], df_meta['asin'], on='asin', how='left', indicator=True, suffixes=("_l", "_r")
)

    print(m.head(5))
    print(len(m[m["_merge"]=="both"]) / len(m))
    print("")

    print("data sample")
    print("features:")
    print(df_meta["description"].head(20))
    print(df_meta["description"][0])