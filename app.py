import streamlit as st
import zipfile
import os
import tempfile
from io import BytesIO
import logging
from main import ReadmeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReadmeGeneratorApp")

UPLOAD_DIR = tempfile.mkdtemp()

st.set_page_config(page_title="Readme Generator", page_icon="📝")


def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def validate_zip_file(file_path):
    """Validate that the uploaded file is a valid zip file."""
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            return True
    except zipfile.BadZipFile:
        return False


def main():
    """Main function to run the Streamlit app."""
    st.title("README Generator")
    st.markdown("Upload your project zip file to generate a professional README.md")

    uploaded_file = st.file_uploader("Upload project .zip file", type="zip")

    if uploaded_file:
        st.info(f"Uploaded: {uploaded_file.name}")

        if not os.path.exists(".env"):
            st.warning("No .env file found. Make sure to set up your API_KEY.")

        saved_file_path = save_uploaded_file(uploaded_file)

        if not validate_zip_file(saved_file_path):
            st.error("Invalid zip file. Please upload a valid zip archive.")
            return

        if st.button("Generate README", type="primary"):
            try:
                with st.spinner("Generating README..."):
                    readme_generator = ReadmeGenerator(zip_file_path=saved_file_path)
                    readme_content = readme_generator.run()

                st.success("README generated successfully!")

                tab1, tab2 = st.tabs(["Preview", "Download"])

                with tab1:
                    st.markdown(readme_content)

                with tab2:
                    md_file = BytesIO()
                    md_file.write(readme_content.encode("utf-8"))
                    md_file.seek(0)

                    st.download_button(
                        label="Download README.md",
                        data=md_file,
                        file_name="README.md",
                        mime="text/markdown",
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error generating README: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
