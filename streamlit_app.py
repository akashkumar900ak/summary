import streamlit as st
from summarizer_module import TextSummarizer
import time
import traceback
import PyPDF2
import tempfile

st.set_page_config(page_title="AI Book & Report Summarizer", page_icon="📘", layout="wide")

if 'summarizer' not in st.session_state:
    st.session_state.summarizer = TextSummarizer()

def extract_text_from_txt(txt_file):
    try:
        return txt_file.read().decode('utf-8').strip()
    except Exception as e:
        st.error(f"❌ Error reading TXT file: {str(e)}")
        return ""

def extract_text_from_pdf_simple(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()]).strip()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return ""

def main():
    st.title("📘 AI Book & Report Summarizer")
    st.markdown("Summarize large documents — **text, tables, and figures** — using fast transformer models.")

    with st.sidebar:
        st.header("⚙️ Settings")
        summary_length = st.selectbox("Summary Length", ["short", "medium", "long"], index=1)
        input_method = st.radio("Input Method", ["Text Input", "File Upload"])
        st.markdown("---")
        st.info("Model: `distilbart-cnn-12-6` (Fast, Abstractive)")

    text_to_summarize = ""
    summary = ""

    if input_method == "Text Input":
        text_to_summarize = st.text_area("📝 Paste your content here:", height=300)
    else:
        uploaded_file = st.file_uploader("📂 Upload a PDF or TXT file", type=["pdf", "txt"])
        if uploaded_file:
            file_size = len(uploaded_file.getbuffer())
            st.info(f"📄 File size: {file_size / 1024:.2f} KB")

            if uploaded_file.type == "text/plain":
                text_to_summarize = extract_text_from_txt(uploaded_file)

            elif uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                st.success("✅ PDF loaded.")
                text_to_summarize = f"__PDF_PATH__:{tmp_path}"

    if st.button("🚀 Generate Summary"):
        if not text_to_summarize or len(text_to_summarize.strip()) < 50:
            st.error("⚠️ Please enter or upload at least 50 characters of content.")
            return

        try:
            summarizer = st.session_state.summarizer
            st.info("⏳ Summarizing content... Please wait.")
            progress = st.progress(0)
            for i in range(50):
                time.sleep(0.01)
                progress.progress(i + 1)

            start = time.time()

            if text_to_summarize.startswith("__PDF_PATH__"):
                pdf_path = text_to_summarize.replace("__PDF_PATH__:", "")
                summary = summarizer.summarize_pdf_with_images(pdf_path, length=summary_length)
            else:
                summary = summarizer.summarize(text_to_summarize, length=summary_length)

            end = time.time()
            progress.empty()

            st.success("✅ Summary Generated!")
            st.markdown("### 📄 Summary:")
            st.write(summary)

            st.markdown("### 📊 Stats:")
            col1, col2, col3 = st.columns(3)
            original_len = len(text_to_summarize)
            summary_len = len(summary)
            col1.metric("Original Length", f"{original_len:,} chars")
            col2.metric("Summary Length", f"{summary_len:,} chars")
            col3.metric("Compression", f"{round((1 - summary_len / original_len) * 100, 1)}%")

            st.caption(f"⏱️ Time taken: {end - start:.2f} seconds")

            st.download_button("📥 Download Summary", summary, file_name="summary.txt", mime="text/plain")

        except Exception as e:
            st.error("❌ An error occurred during summarization.")
            with st.expander("Details"):
                st.code(traceback.format_exc())

    else:
        st.info("👆 Paste text or upload a file to begin.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit + Hugging Face<br>
        <a href='https://streamlit.io/cloud' target='_blank'>Deploy your own app for free</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 App crashed.")
        st.exception(e)
