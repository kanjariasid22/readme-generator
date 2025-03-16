# README Generator

![README Generator](https://img.shields.io/badge/README-Generator-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange)

A powerful tool to automatically generate comprehensive README.md files by analyzing your project's codebase. Using Google's Generative AI (Gemini), this tool creates well-structured documentation that includes project descriptions, installation instructions, usage guidelines, and more - all with a simple file upload.

## 🚀 Features

- **Simple Upload Interface**: Just upload your project as a .zip file
- **AI-Powered Analysis**: Leverages Google's Gemini model to understand your codebase
- **Comprehensive Documentation**: Generates complete README.md with all essential sections:
  - Project description and overview
  - Feature lists
  - Installation instructions
  - Usage examples
  - Project structure
  - Dependencies
  - License recommendations
- **Instant Preview**: Review the generated README before downloading
- **Multiple Language Support**: Works with Python, JavaScript, Java, C/C++, HTML/CSS, and more
- **Customizable Output**: Download the README and make any final adjustments

## 📋 Requirements

- Python 3.8 or higher
- Google API key for Gemini model access

## 🔧 Installation

### Clone the repository:

```bash
git clone git@github.com:kanjariasid22/readme-generator.git
cd readme-generator
```

### Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Set up environment variables:

1. Create a `.env` file in the root directory
2. Add your Google API key to the `.env` file:
   ```
   API_KEY=your_google_api_key_here
   ```

## 🚀 Usage

### Start the application:

```bash
streamlit run app.py
```

### Generate a README:

1. Open your browser and navigate to the displayed URL (typically http://localhost:8501)
2. Upload your project's .zip file using the file uploader
3. Click the "Generate README" button
4. Wait for the analysis to complete
5. Preview the generated README in the "Preview" tab
6. Download the README.md file from the "Download" tab
7. Add the file to your project repository

## 🧩 How It Works

1. **File Analysis**: The tool extracts and processes files from your .zip archive
2. **Code Summarization**: Each file is analyzed to understand its purpose and functionality
3. **Content Generation**: The AI model creates appropriate documentation sections
4. **README Structuring**: Information is organized into a standard README format

## 📁 Project Structure

- **`app.py`**: Streamlit web interface for uploading files and displaying results
- **`main.py`**: Core functionality for extracting files and generating README content
- **`requirements.txt`**: Python dependencies
- **`.env`**: Environment variables configuration

## 🔄 Dependencies

- **Streamlit**: Web interface framework
- **LangChain**: AI orchestration for document processing
- **Google Generative AI**: Gemini model for README generation
- **python-dotenv**: Environment variable management
