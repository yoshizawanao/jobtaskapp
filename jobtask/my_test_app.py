from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
result = llm.invoke("こんにちは！ChatGpt！")
print(result)  

