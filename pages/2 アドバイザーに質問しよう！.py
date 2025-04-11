import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler

# models
from langchain_openai import ChatOpenAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

import os

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "jobtaskapp"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🧐"
    )
    st.sidebar.title("Options")

def select_model():
    temperature= st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0,step=0.01
    )
    models = ("GPT-4", "GPT-3.5 (not recommended)")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5 (not recommended)":
        return ChatOpenAI(
            temperature=temperature, model_name="gpt-3.5-turbo")
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature, model_name="gpt-4o")



def init_qa_chain(llm):
    llm = llm
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
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
       
        search_type="similarity",

        search_kwargs={"k":10}
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain





CUSTOM_SYSTEM_PROMPT2 = """
あなたは、ユーザーのリクエストに基づいた観光地をインターネットで調べ提供するアシスタントです。
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
    
def page_ask_my_pdf(llm, prompt):
    chain = init_qa_chain(llm)
    # st.markdown("## Answer")
    # answer_container = st.empty()
    full_response = chain.invoke(prompt)
    
    st.session_state['memory'].save_context(
        {"input": prompt + "(パンフレット参照)"},
        {"output": full_response}
    )   
    return full_response

def create_agent(llm):
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT2),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = llm
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
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
    llm=select_model()
    st.title("アドバイザーに質問しよう！ 🧐")
    if "vectorstore" not in st.session_state:
        st.warning("まずは 📄 Upload PDF(s) からPDFファイルをアップロードしてね")
        return
    
    with st.chat_message("ai"):
        st.markdown("""
                    こんにちは！私は観光地アドバイザーです！  
                    あなたが行きたい観光地の条件を教えていただけたら、  
                    1.パンフレットを参照して提案(履歴に(パンフレット参照)と表記されるよ！)   
                    2.インターネット検索より提案  
                    の２つの方法であなたに適した観光地を提案させていただきます！  
                    適切な提案をするために、必ず質問には[場所]を記入してください。    
                    出来るだけ細かい質問をしてくれるとより具体的な提案が可能です！  
                    """)
    
    web_browsing_agent = create_agent(llm)

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="湘南の家族連れにおすすめの観光地は？"):
        st.chat_message("user").write(prompt + "(パンフレット参照)")
        # page_ask_my_pdf(llm, prompt)

        with st.chat_message("assistant"):
            answer = page_ask_my_pdf(llm, prompt)
            st.write(answer)

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