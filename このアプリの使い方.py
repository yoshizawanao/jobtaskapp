import streamlit as st


def init_page():
    st.set_page_config(
        page_title="Ask AI tour-adviser",
        page_icon="🧐"
    )
    st.header("Ask AI tour-adviser 🧐")


def main():
    init_page()

    st.sidebar.success("👆のメニューから進んでね")

    st.markdown(
    """
    ### Ask AI tour-adviser にようこそ！

    - このアプリは、あなたにあった観光地を提案するアプリです。
    - 行先の観光地のパンフレット(PDF)を送信した後、あなたが行きたい観光地の条件を提示すると自動でAIがあなたにあった観光地を提案してくれます。
    - まずは左のメニューから `Upload PDF` を選択して観光地のパンフレット(PDF)をアップロードしてください。
    - PDFをアップロードしたら `アドバイザーに質問しよう！` を選択して質問をしてみましょう😇
    """
    
    )

if __name__ == '__main__':
    main()