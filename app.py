import validators, streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
import openai
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Create a Streamlit app

st.set_page_config(page_title="Langchain : Summarize Text from Yt or Website", page_icon="$")
st.title("Langchain : Summarize Text from YT or Website")
st.subheader("Summarize URL")

# Get the Groq Api Key 
with st.sidebar:
  groq_api_key = st.text_input("Groq_API_Key", value = "", type="password")

# LLM model 
llm = ChatGroq(model_name = "openai/gpt-oss-20b", api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
content:{text}
"""

prompt = PromptTemplate(input_variables=["text"], template=prompt_template)


URL = st.text_input("URL", label_visibility= "collapsed")


if st.button("Summarize the content from YT or Website"):
  if not groq_api_key.strip() or not URL.strip():
    st.error("Please provide the information to get started")
  elif not validators.url(URL):
    st.error("Please enter the valid URL.It can be a YT or Website URL")
  else:
    try:
      with st.spinner("Waiting..."):
        #loading the data
        if "youtube.com" in URL:
          loader = YoutubeLoader.from_youtube_url(URL, add_video_info = True)
        else:
          loader = UnstructuredURLLoader(urls=[URL], ssl_verify =False,
                                         headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
        docs  = loader.load()

        #chain for summarization 
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        output_summary = chain.invoke(docs)

        st.success(output_summary)
    except Exception as e:
      st.exception(e)


