import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

### Function to get response from LLMA 2 model

def get_llma_response(input_text, no_words, blogs_style):

    #### LLMA2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q3_K_s.bin',
                        model_type='llama',
                        # config={'man_new_tokens': 256,
                        #         'tempreture': 0.01}
                        )
    ## Prompt Template
    template = f"""
    Write a blog for {blogs_style} job profile a topic {input_text}
    within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=['blogs_style', 'input_text', 'no_words'],
                            template=template)

    ## Generate the response from the LLMA model
    response = llm(prompt.format(blogs_style=blogs_style, input_text=input_text,
                      no_words = no_words))

    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs")

input_text = st.text_input("Topic:::")

## Creating two more columns for additional 2 fields

col1, col2 = st.columns([5,5])


with col1:
    no_words=st.text_input("No of words:::")

with col2:
    blogs_style = st.selectbox('Writing the blog for ',
                               ('Researchers', 'Data Scientist', 'Common people'), index = 0)

submit = st.button("Generate")

## Final response

if submit:
    st.write(get_llma_response(input_text, no_words, blogs_style))

