import streamlit as st
from gensim.models import KeyedVectors, Word2Vec
import os
import re
import chardet
from sudachipy import tokenizer
from sudachipy import dictionary
import numpy as np
import pandas as pd

# カスタムCSSでフォントサイズを調整
st.markdown(
    """
    <style>
    body {
        font-size: 16px !important;
    }
    .stButton button {
        font-size: 16px !important;
    }
    .stTextInput input {
        font-size: 16px !important;
    }
    .stSelectbox div, .stMultiSelect div {
        font-size: 16px !important;
    }
    .stFileUploader div {
        font-size: 16px !important;
    }
    .stTable th, .stTable td {
        font-size: 16px !important;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# モデルのロード関数
@st.cache_resource
def load_model(path):
    try:
        # バイナリ形式で読み込む
        return KeyedVectors.load_word2vec_format(path, binary=True)
    except Exception as binary_error:
        try:
            # テキスト形式で再試行
            return KeyedVectors.load_word2vec_format(path, binary=False)
        except Exception as text_error:
            raise ValueError(f"モデルの読み込みに失敗しました:\nバイナリ形式エラー: {binary_error}\nテキスト形式エラー: {text_error}")

# ファイルの文字コードを判定して読み込む関数
def read_file_with_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        return raw_data.decode(detected_encoding)

# 分かち書き処理関数
def tokenize_text(text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    sentences = []
    for sentence in text.splitlines():
        if sentence.strip():
            words = [m.surface() for m in tokenizer_obj.tokenize(sentence, mode)
                     if m.part_of_speech()[0] not in ["助詞", "助動詞"] and not re.match(r'[\(\)\[\]\{\}]', m.surface())]
            if words:
                sentences.append(words)
    return sentences

# 青空文庫からモデルを作成する関数
def train_model_from_texts(text_paths, model_path):
    sentences = []
    for text_path in text_paths:
        if not os.path.exists(text_path):
            st.error(f"テキストファイルが見つかりません: {text_path}")
            continue

        text = read_file_with_encoding(text_path)
        sentences.extend(tokenize_text(text))

    if not sentences:
        st.error("有効なテキストがありませんでした。モデルを作成できません。")
        return None

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format(model_path, binary=True)
    return model.wv

# 初期化
model_paths = {
    "Wikipediaモデル": './wikipedia/entity_vector.model.bin',
}
if 'custom_models' not in st.session_state:
    st.session_state['custom_models'] = {}

if 'history' not in st.session_state:
    st.session_state['history'] = {}

if 'search_results' not in st.session_state:
    st.session_state['search_results'] = None

if 'weights' not in st.session_state:
    st.session_state['weights'] = {}

# サイドバーでモードを選択
st.sidebar.title("モード選択")
mode = st.sidebar.radio("操作を選んでください:", ["検索", "新しいモデルの作成", "既存モデルの読み込み", "モデルの削除"])

if mode == "検索":
    st.title("類義語検索")

    # 複数モデル選択
    model_options = list(model_paths.keys()) + list(st.session_state['custom_models'].keys())
    selected_models = st.multiselect("使用するモデルを選択してください", model_options, default=model_options)

    # 検索方法
    aggregation_method = st.selectbox("検索結果の並び順:", ["平均値順", "最大値順"])

    # 結果トークン数
    topn = st.number_input("表示するトークン数", min_value=1, max_value=50, value=10, step=1)

    # 入力フィールド
    positive_text = st.text_input("ポジティブ単語 (例: イチロー, サッカー)", "")
    negative_text = st.text_input("ネガティブ単語 (例: 野球)", "")

    # 選択されたモデルをロード
    models = {}
    for model_name in selected_models:
        if model_name in model_paths:
            models[model_name] = load_model(model_paths[model_name])
        elif model_name in st.session_state['custom_models']:
            models[model_name] = st.session_state['custom_models'][model_name]

    # 検索ボタン
    search_clicked = st.button("検索")

    if search_clicked:
        if not models:
            st.error("選択されたモデルがロードされていません。")
        elif positive_text.strip() == "" and negative_text.strip() == "":
            st.warning("ポジティブまたはネガティブ単語を入力してください。")
        else:
            try:
                positive_words = [word.strip() for word in re.split(r'[，,、\s\u3000]+', positive_text) if word.strip()]
                negative_words = [word.strip() for word in re.split(r'[，,、\s\u3000]+', negative_text) if word.strip()]

                # 結果を格納
                combined_results = {}
                for model_name, model in models.items():
                    try:
                        results = model.most_similar(positive=positive_words, negative=negative_words, topn=topn)
                        combined_results[model_name] = results
                    except KeyError as e:
                        st.warning(f"モデル {model_name} に存在しない単語があります: {e}")

                st.session_state['search_results'] = combined_results

                # 初期の重みを設定
                if not st.session_state['weights']:
                    for model_name in combined_results.keys():
                        st.session_state['weights'][model_name] = 1.0

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

    # 個別モデルの結果表示
    if st.session_state['search_results']:
        st.write("### モデル別検索結果")
        for model_name, results in st.session_state['search_results'].items():
            st.write(f"#### {model_name}")
            model_df = pd.DataFrame(results, columns=["単語", "類似度"])
            st.table(model_df)

        # 重み設定
        st.write("### モデルの重み設定")
        weights = {}
        for model_name in st.session_state['search_results'].keys():
            # 重みを設定
            weights[model_name] = st.number_input(
                f"{model_name} の重み",
                min_value=0.0,
                value=st.session_state['weights'].get(model_name, 1.0),
                step=0.1,
                key=f"weight_{model_name}"
            )

            # 重み変更時に加重平均計算を更新
            st.session_state['weights'][model_name] = weights[model_name]

        # 加重平均計算と表示
        st.write("### 総合順位")
        weighted_results = {}
        for model_name, results in st.session_state['search_results'].items():
            for word, similarity in results:
                if word not in weighted_results:
                    weighted_results[word] = 0
                weighted_results[word] += similarity * st.session_state['weights'][model_name]

        # ソートして表示
        sorted_results = sorted(weighted_results.items(), key=lambda x: x[1], reverse=True)
        total_df = pd.DataFrame(sorted_results, columns=["単語", "加重平均スコア"])
        st.table(total_df)

elif mode == "新しいモデルの作成":
    st.title("新しいモデルの作成")
    text_files = st.file_uploader("青空文庫のテキストファイルを複数アップロードしてください", type=["txt"], accept_multiple_files=True)
    new_model_name = st.text_input("新しいモデルの名前を入力してください")

    if st.button("モデルを作成"):
        if text_files and new_model_name.strip():
            temp_paths = []
            for text_file in text_files:
                temp_path = f"./{text_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(text_file.read())
                temp_paths.append(temp_path)

            model_path = f"./{new_model_name}.model.bin"
            trained_model = train_model_from_texts(temp_paths, model_path)
            if trained_model:
                st.session_state['custom_models'][new_model_name] = trained_model
                model_paths[new_model_name] = model_path
                st.success(f"モデル {new_model_name} を作成して登録しました！")
            for temp_path in temp_paths:
                os.remove(temp_path)
        else:
            st.warning("テキストファイルとモデル名を入力してください。")

elif mode == "既存モデルの読み込み":
    st.title("既存モデルの読み込み")
    existing_model_file = st.file_uploader("既存のWord2Vecモデルファイルをアップロードしてください", type=["bin"])

    if st.button("モデルを読み込む"):
        if existing_model_file:
            temp_path = f"./{existing_model_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(existing_model_file.read())
            try:
                loaded_model = load_model(temp_path)
                st.session_state['custom_models'][existing_model_file.name] = loaded_model
                model_paths[existing_model_file.name] = temp_path
                st.success(f"モデル {existing_model_file.name} を読み込みました！")
            except ValueError as e:
                st.error(f"モデルの読み込みに失敗しました: {e}")
                os.remove(temp_path)
            except Exception as e:
                st.error(f"予期しないエラーが発生しました: {e}")
                os.remove(temp_path)
        else:
            st.warning("モデルファイルをアップロードしてください。")

elif mode == "モデルの削除":
    st.title("モデルの削除")
    delete_model_name = st.selectbox("削除するモデルを選択してください", list(st.session_state['custom_models'].keys()))

    if st.button("モデルを削除"):
        if delete_model_name in st.session_state['custom_models']:
            del st.session_state['custom_models'][delete_model_name]
            model_file_path = model_paths.pop(delete_model_name, None)
            if model_file_path and os.path.exists(model_file_path):
                os.remove(model_file_path)
            st.success(f"モデル {delete_model_name} を削除しました！")
        else:
            st.error("選択されたモデルは存在しません。")
