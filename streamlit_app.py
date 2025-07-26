import streamlit as st
from summarizer_module import TextSummarizer
import time
import PyPDF2
import traceback

st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide"
)

if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None

def load_summarizer():
    if st.session_state.summarizer is None:
        with st.spinner("Loading model..."):
            st.session_state.summarizer = TextSummarizer()
    return st.session_state.summarizer

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()]).strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    try:
        return txt_file.read().decode('utf-8').strip()
    except Exception as e:
        st.error(f"Error reading TXT file: {str(e)}")
        return None

def main():
    st.title("📘 Ultra-Long Content Summarizer")
    st.markdown("Summarize long documents like **books, reports, or articles** – even up to 100 million characters!")

    with st.sidebar:
        st.header("⚙️ Settings")
        summary_length = st.selectbox("Summary Length", ["short", "medium", "long"], index=1)
        input_method = st.radio("Input Method", ["Text Input", "File Upload"])
        st.markdown("### 💡 Tips")
        st.markdown("- Minimum: 50 characters\n- Max: 100,000,000 (chunked)\n- PDF/TXT supported")

    summary = ""
    text_to_summarize = ""

    # INPUT SECTION
    if input_method == "Text Input":
        text_to_summarize = st.text_area("📥 Paste your text:", height=300)
    else:
        uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["txt", "pdf"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text_to_summarize = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text_to_summarize = extract_text_from_txt(uploaded_file)

            if text_to_summarize:
                st.success(f"✅ Loaded file: {len(text_to_summarize):,} characters")
                with st.expander("📄 Preview"):
                    st.text_area("File preview", text_to_summarize[:3000] + "..." if len(text_to_summarize) > 3000 else text_to_summarize, height=200)

    # LIMIT CHECK
    if len(text_to_summarize) > 100_000_000:
        st.error("❌ Content exceeds the 100,000,000 character limit.")
        return

    # WARN FOR LARGE TEXTS
    if len(text_to_summarize) > 500_000:
        st.warning("⚠️ Large text detected. This may take several minutes to process on free-tier servers.")

    # SUMMARIZE
    if st.button("🚀 Generate Summary"):
        if len(text_to_summarize.strip()) < 50:
            st.error("⚠️ Minimum 50 characters required.")
        else:
            try:
                summarizer = load_summarizer()
                with st.spinner("Summarizing, please wait..."):
                    start = time.time()
                    summary = summarizer.summarize(text_to_summarize, length=summary_length)
                    duration = time.time() - start

                st.success("✅ Summary Generated!")
                st.markdown("### 📋 Summary:")
                st.write(summary)

                st.markdown("### 📊 Stats:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Original", f"{len(text_to_summarize):,} chars")
                col2.metric("Summary", f"{len(summary):,} chars")
                compression = round((1 - len(summary) / len(text_to_summarize)) * 100, 1)
                col3.metric("Compression", f"{compression}%")
                st.caption(f"⏱️ Time taken: {duration:.2f} sec")

                st.download_button(
                    "📥 Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error("❌ An error occurred during summarization.")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Paste or upload content and click 'Generate Summary'.")

if __name__ == "__main__":
    main()
