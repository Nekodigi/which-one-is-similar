from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import vertexai
from PIL import Image as PILImage
from streamlit.runtime.uploaded_file_manager import UploadedFile
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)

PROJECT_ID = "ordinal-virtue-418309"
LOCATION = "asia-northeast1"


def get_image_embeddings(
    file: UploadedFile,
) -> MultiModalEmbeddingResponse:

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    embeddings = model.get_embeddings(
        image=Image(file.getvalue()),
    )
    return embeddings


st.title("この画像はどっちに似てる？")
st.markdown(
    """
    2個の画像の中で、アップロードした画像がどちらに近いか判定するWebアプリです  
    既存の画像認識サービスは使っておらず、画像をより小さな特徴量に変換して比較しています"""
)

st.header("入力", divider="rainbow")
st.markdown(
    """
    Browse filesをクリックして3つの画像をアップロードしてください  
    しばらくすると結果が表示されます"""
)
col1, col2 = st.columns(2)
embed1, embed2, embed3 = None, None, None

with col1:
    st.subheader("画像1")
    image1 = st.file_uploader("画像をアップロード(例)手", type=["png", "jpg", "jpeg"])
    if image1 is not None:
        st.image(image1, caption="画像1", width=200)
        embed1 = np.array(get_image_embeddings(image1).image_embedding)
with col2:
    st.subheader("画像2")
    image2 = st.file_uploader(
        "画像をアップロード(例)パソコン", type=["png", "jpg", "jpeg"]
    )
    if image2 is not None:
        st.image(image2, caption="画像2", width=200)
        embed2 = np.array(get_image_embeddings(image2).image_embedding)
st.subheader("比べたい画像")
image3 = st.file_uploader(
    "画像をアップロード(例)先ほどの画像1/2のどちらかに似た物",
    type=["png", "jpg", "jpeg"],
)
if image3 is not None:
    st.image(image3, caption="画像3", width=200)
    embed3 = np.array(get_image_embeddings(image3).image_embedding)

st.header("結果", divider="rainbow")


if (
    isinstance(embed1, np.ndarray)
    and isinstance(embed2, np.ndarray)
    and isinstance(embed3, np.ndarray)
):
    dist1 = np.linalg.norm(embed1 - embed3)
    dist2 = np.linalg.norm(embed2 - embed3)
    sum_dist = dist1 + dist2
    if dist1 < dist2:
        st.metric(
            "似てる度", f"{int(dist2/sum_dist*100)}%画像1", f"ベクトルの距離{dist2:.2e}"
        )
    else:
        st.metric(
            "似てる度", f"{int(dist1/sum_dist*100)}%画像2", f"ベクトルの距離{dist2:.2e}"
        )

st.header("このサイトについて", divider="rainbow")
st.markdown(
    """
            [AI実践講座](https://techno-semi.com/practice)の一環で作成したアプリです  
            開発で使用した資料をまとめたスライドやソースコードを公開しています  
            - [Google Slide](https://docs.google.com/presentation/d/1pnTyeNN3i1GLaQXUVIyQCovqAw-wRSNx/edit?usp=sharing&ouid=101875186374769306062&rtpof=true&sd=true)  
            - [Source Code(このWebサイト)](https://streamlit-gcp-aohlqyvu3a-an.a.run.app)  
            - [Source Code(コンペ)](https://github.com/Nekodigi/Lightning-Lab)
            
            今回の開発にあたり、これまで積み重ねてきたWeb開発、アート制作、AI開発、言語理解の知識が大きく活かされました  
            数百以上のアプリのソースコード、開発秘話を公開しているので是非筆者のホームページも覗いてみてください
            - [Homepage](https://nekodigi.com/)
            - [Youtube](https://www.youtube.com/c/Nekodigi)
            """
)
