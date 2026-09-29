"""Streamlit UI for the AI Resume Feedback Generator."""

import streamlit as st

from resume_feedback import ResumeFeedbackError, extract_resume_text, generate_feedback

st.set_page_config(page_title="AI Resume Feedback Generator", page_icon="📄", layout="centered")
st.title("📄 AI Resume Feedback Generator")
st.write("Get evidence-based feedback on how your resume aligns with a job description.")
st.info(
    "Privacy: your resume and job description are sent to Anthropic's API when you generate feedback. "
    "Do not upload documents unless you are comfortable sharing their contents with that service."
)

uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Paste the job description", height=220, max_chars=20_000)
consent = st.checkbox("I understand my resume and job description will be sent to Anthropic for processing.")

if st.button("Generate feedback", type="primary", disabled=not (uploaded and job_description.strip() and consent)):
    with st.spinner("Reviewing your resume..."):
        try:
            resume_text = extract_resume_text(uploaded.name, uploaded.getvalue())
            if not resume_text:
                st.error("No readable text was found. Try a text-based PDF, DOCX, or TXT file.")
            else:
                feedback = generate_feedback(resume_text, job_description.strip())
                st.markdown(feedback)
                st.download_button("Download feedback as Markdown", data=feedback,
                                   file_name="resume-feedback.md", mime="text/markdown")
        except ResumeFeedbackError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Something unexpected happened. Please try again.")
