import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🧐"
    )
    st.sidebar.title("Options")

def select_model():
    models = ("GPT-4", "GPT-3.5 (not recommended)")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5 (not recommended)":
        return ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo")
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=0, model_name="gpt-4o")



def init_qa_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template("""
    あなたはユーザーの求める条件にあった観光地を提案するアシスタントです。
    以下の前提知識(観光地のパンフレット)を用いて、ユーザーからの質問に答えてください。

    ===
    前提知識
    {context}

    ===
    ユーザーからの質問
    {question}
    """)
    retriever = st.session_state.vectorstore.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":10}
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def page_ask_my_pdf():
    chain = init_qa_chain()

    if query := st.text_input("PDFへの質問を書いてね: ", key="input"):
        st.markdown("## Answer")
        st.write_stream(chain.stream(query))




# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/main.py
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler


# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

CUSTOM_SYSTEM_PROMPT = """
あなたは、ユーザーのリクエストに基づいてインターネットで調べ物を行うアシスタントです。
利用可能なツールを使用して、調査した情報を説明してください。
既に知っていることだけに基づいて答えないでください。回答する前にできる限り検索を行ってください。
(ユーザーが読むページを指定するなど、特別な場合は、検索する必要はありません。)

検索結果ページを見ただけでは情報があまりないと思われる場合は、次の2つのオプションを検討して試してみてください。

- 検索結果のリンクをクリックして、各ページのコンテンツにアクセスし、読んでみてください。
- 1ページが長すぎる場合は、3回以上ページ送りしないでください（メモリの負荷がかかるため）。
- 検索クエリを変更して、新しい検索を実行してください。
- 検索する内容に応じて検索に利用する言語を適切に変更してください。
  - 例えば、プログラミング関連の質問については英語で検索するのがいいでしょう。

ユーザーは非常に忙しく、あなたほど自由ではありません。
そのため、ユーザーの労力を節約するために、直接的な回答を提供してください。

=== 悪い回答の例 ===
- これらのページを参照してください。
- これらのページを参照してコードを書くことができます。
- 次のページが役立つでしょう。

=== 良い回答の例 ===
- これはサンプルコードです。 -- サンプルコードをここに --
- あなたの質問の答えは -- 回答をここに --

回答の最後には、参照したページのURLを**必ず**記載してください。（これにより、ユーザーは回答を検証することができます）

ユーザーが使用している言語で回答するようにしてください。
ユーザーが日本語で質問した場合は、日本語で回答してください。ユーザーがスペイン語で質問した場合は、スペイン語で回答してください。
"""




def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは！なんでも質問をどうぞ！"}
        ]
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )

        # このようにも書ける
        # from langchain_community.chat_message_histories import StreamlitChatMessageHistory
        # msgs = StreamlitChatMessageHistory(key="special_app_key")
        # st.session_state['memory'] = ConversationBufferMemory(memory_key="history", chat_memory=msgs)



    

def create_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )


def main():
    init_page()
    init_messages()
    st.title("PDF QA 🧐")
    if "vectorstore" not in st.session_state:
        st.warning("まずは 📄 Upload PDF(s) からPDFファイルをアップロードしてね")
    else:
        page_ask_my_pdf()
    web_browsing_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="2023 FIFA 女子ワールドカップの優勝国は？"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # コールバック関数の設定 (エージェントの動作の可視化用)
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)

            # エージェントを実行
            response = web_browsing_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])



if __name__ == '__main__':
    main()