import validators
import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub

# ----------------------------
# Streamlit UI Configuration
# ----------------------------
st.set_page_config(page_title="GenAI Summarizer", page_icon="📄", layout="wide")
st.title("🎥 GenAI Summarizer: YouTube & Website Content")

with st.sidebar:
    st.subheader("🔑 API Settings")
    Hf_api_key = st.text_input("Enter HuggingFace API Key", type="password")

# Input URL
URL = st.text_input("Paste a YouTube or Website URL below 👇")

# Summary options
summary_style = st.radio(
    "Choose summary style:",
    ["Concise (150 words)", "Standard (300 words)", "Detailed (500 words)"]
)

word_limit = {
    "Concise (150 words)": 150,
    "Standard (300 words)": 300,
    "Detailed (500 words)": 500,
}[summary_style]

# ----------------------------
# LLM Setup (FIXED)
# ----------------------------
# Mixtral CANNOT run on HuggingFaceHub API → replaced with a working summarization model
Repo_id = "facebook/bart-large-cnn"

if Hf_api_key:
    llm = HuggingFaceHub(
        repo_id=Repo_id,
        huggingfacehub_api_token=Hf_api_key,
        task="text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 600
        }
    )
else:
    llm = None


# ----------------------------
# Main Summarization Logic
# ----------------------------
if st.button("🚀 Summarize"):
    if not Hf_api_key:
        st.error("Please enter your Hugging Face API Key.")
    elif not URL:
        st.error("Please enter a YouTube or Website URL.")
    elif not validators.url(URL):
        st.error("Invalid URL format.")
    else:
        try:
            with st.spinner("⏳ Fetching content..."):

                # Detect content type
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
                    st.stop()

            with st.spinner("📘 Cleaning and preparing text..."):
                # Split to avoid token overflow
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

            with st.spinner("🤖 Generating summary..."):
                
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce"
                )

                # FIX: map_reduce chain DOES NOT accept word_limit
                summary = chain.run({"input_documents": chunks})

            st.success("✅ Summary Generated!")
            st.write(summary)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("If this is a YouTube link, ensure subtitles are available.")













