import streamlit as st
import requests
import time


st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


st.title('Web RAG with Mistral 7B')


def decode_stream(r):
    """Decode each stream yield as utf-8

    Args:
        r (generator of str): Generator of bytes strings

    Yields:
        str: Each decoded string
    """
    for chunk in r:
        yield chunk.decode('utf-8')


## Get user query
if query := st.chat_input('Enter your query'):
    with st.chat_message('user'):
        st.markdown(query)

    ## Get bot answer
    with st.chat_message('assistant'):
        start_time = time.time()
        with requests.post('http://web-rag-backend:8080/search', json={'content': query}, stream=True) as r:
            for line in r.iter_lines():
                line_str = line.decode('utf-8')
                if line_str == 'Answer:':
                    ## Once the bot answer starts, switch to standard font
                    st.markdown(f'---')
                    st.markdown(f'**Answer:**')
                    st.write_stream(decode_stream(r))
                    break
                else:
                    ## Indicate progress steps as blue and italic text
                    st.markdown(f'*:blue[{line_str}]*')

        ## Indicate elapsed time for the request
        elapsed_time = time.time() - start_time
        st.markdown(f'---')
        st.markdown(f'*:red[- Time elapsed {elapsed_time:0.2f}s.]*')