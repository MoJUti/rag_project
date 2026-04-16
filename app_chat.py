import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as  st
from uuid import uuid4
from rag import RagService
from file_history_store import get_history
import config_data as config
import time

# 标题
st.title("智能客服")
# 分隔符
st.divider()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"chat_{uuid4().hex[:12]}"

session_config = config.build_session_config(st.session_state["session_id"])

# 避免性能压力，session_state 存入对象
if "message" not in st.session_state:
    # 从本地文件加载该 session_id 对应的历史记录
    history = get_history(st.session_state["session_id"])
    
    messages = []
    # 遍历已保存的聊天历史
    for msg in history.messages:
        # LangChain 中的 type 对应 human/ai，我们映射成 streamlit 的 user/assistant
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
        
    # 如果本地没有历史记录，则给出一个默认欢迎语
    if not messages:
        messages = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
        
    st.session_state["message"] = messages

if "rag" not in st.session_state:
    st.session_state["rag"]= RagService()
#循环 输出历史信息，原本只记录但页面不显示
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
# 在页面最下方提供用户输入栏
prompt= st.chat_input()

if prompt :
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list= []
    with st.spinner("AI 思考中......."):
        # 调用 RAG服务

        # 直接输出
        # res= st.session_state["rag"].chain.invoke({"input":prompt},session_config)
        #
        # st.chat_message("assistant").write(res)
        # st.session_state["message"].append({"role":"assistant","content":res})

        # 流式输出
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

