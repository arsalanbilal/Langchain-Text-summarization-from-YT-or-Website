import validators
import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit page setup
st.set_page_config(page_title="GenAI Summarizer", page_icon="📄")
st.title("🎥 LangChain: Summarize YouTube or Website Content")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("🔑 Enter your Groq API Key", type="password")

# Input URL
URL = st.text_input("Paste a YouTube or Website URL below 👇")

# LLM setup
if groq_api_key:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)
else:
    llm = None

prompt_template = """
Summarize the following content in about 300 words, highlighting key insights and main points:
{text}
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

if st.button("🚀 Summarize"):
    if not groq_api_key.strip() or not URL.strip():
        st.error("Please enter both API key and URL.")
    elif not validators.url(URL):
        st.error("Invalid URL format.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Handle both YouTube and website URLs
                if "youtube.com" in URL or "youtu.be" in URL:
                    loader = YoutubeLoader.from_youtube_url(URL, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[URL],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                docs = loader.load()
                if not docs:
                    st.error("Couldn't fetch content. Try another URL.")
                else:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)
                    st.success(summary)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("If this is a YouTube link, make sure the video has subtitles enabled.")







