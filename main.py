import logging
import os
import shutil
import zipfile
from typing import List, Optional
from dotenv import load_dotenv

from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
API_KEY = os.environ.get("API_KEY")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ReadmeGenerator")

INCLUDE_EXTENSIONS = [
    "py",
    "java",
    "html",
    "css",
    "c",
    "js",
    "ipynb",
    "cpp",
    "sh",
    "md",
    "json",
    "yml",
    "yaml",
    "xml",
    "go",
    "rb",
    "php",
    "ts",
    "jsx",
    "tsx",
]


class LLMSummarize:
    """Class to handle all LLM summarization operations."""

    def __init__(self, api_key: str):
        """
        Initialize the LLM summarization class.

        Args:
            api_key (str): API key for Google Generative AI.

        Raises:
            ValueError: If API key is not provided.
        """
        if not api_key:
            raise ValueError("API_KEY is not set in the environment or .env file.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", google_api_key=api_key
        )

        self.code_summary_prompt = """You are a code summarization assistant. Analyze the provided code snippet and provide a concise summary of its functionality. Include:
                                        - The purpose of the code.
                                        - Key features or methods.
                                        - Dependencies and external libraries used.
                                        
                                        Code snippet: {context}"""

        self.readme_generation_prompt = """You are a technical documentation expert. Using the summaries of individual code files, create a comprehensive README.md file for this project.

                                        Format your response in clear markdown with the following specific sections:

                                        # Project Name
                                        (Infer an appropriate name based on the code)

                                        A concise overview of what this project does, its purpose, and its core functionality.

                                        ## Features
                                        * List the main capabilities and features of the project
                                        * Describe what problems it solves
                                        * Highlight any unique or important aspects

                                        ## Installation and Setup
                                        Provide clear step-by-step instructions for:
                                        * Installing the project
                                        * Configuring environment variables (especially API keys)
                                        * Setting up any necessary prerequisites

                                        ## Usage
                                        * Show practical examples of how to use the main functionality
                                        * Include sample code snippets where appropriate
                                        * Explain key parameters and options

                                        ## Project Structure
                                        * Outline the organization of the codebase
                                        * Explain the purpose of main files, classes, and directories
                                        * Show how components relate to each other

                                        ## Dependencies
                                        * List all external libraries and frameworks used
                                        * Explain how to install or update dependencies
                                        * Note any version requirements or compatibility issues

                                        ## License
                                        Suggest an appropriate open-source license for this type of project.

                                        ## Contributing
                                        Add a brief section encouraging contributions and explaining how others can contribute.

                                        Make the README professional, accurate and helpful. Base all information exclusively on the provided code summaries:

                                        {context}"""

    def summarize_repo(self, code_list: List[str]) -> str:
        """
        Summarize the repository based on the provided code files.

        Args:
            code_list (List[str]): List of code file contents.

        Returns:
            str: Generated README content.
        """
        if not code_list:
            logger.error("Empty code_list provided.")
            return "No code files found for summarization."

        documents = [
            Document(page_content=code, metadata={"source": f"file_{i}"})
            for i, code in enumerate(code_list)
        ]
        logger.info(f"Prepared {len(documents)} documents for processing.")

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=5000, chunk_overlap=500
        )
        split_docs = text_splitter.split_documents(documents)
        if not split_docs:
            logger.error("Text splitting failed. No chunks created.")
            return "Failed to split documents for processing."

        logger.info(f"Split into {len(split_docs)} chunks for processing.")

        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        map_prompt = PromptTemplate.from_template(self.code_summary_prompt)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        reduce_prompt = PromptTemplate.from_template(self.readme_generation_prompt)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_prompt=document_prompt,
            document_variable_name="context",
        )

        collapse_prompt = PromptTemplate.from_template(
            "Summarize this content for use in README generation: {context}"
        )
        collapse_chain = LLMChain(llm=self.llm, prompt=collapse_prompt)
        collapse_documents_chain = StuffDocumentsChain(
            llm_chain=collapse_chain,
            document_prompt=document_prompt,
            document_variable_name="context",
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=collapse_documents_chain,
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",
            return_intermediate_steps=True,
        )

        try:
            logger.info("Running MapReduce chain to generate README.")
            result = map_reduce_chain.invoke(split_docs)

            output_text = result.get("output_text", "No output text received.")
            if not output_text:
                logger.error("MapReduce chain produced no output.")
                return "Failed to generate README content."

            logger.info("README generated successfully.")
            return output_text
        except Exception as e:
            logger.error(f"Error in MapReduce chain: {str(e)}")
            return f"An error occurred during README generation: {str(e)}"


class ReadmeGenerator:
    """
    Main class to handle repository analysis and README generation.
    """

    def __init__(self, zip_file_path: str):
        """
        Initialize the README generator.

        Args:
            zip_file_path (str): Path to the uploaded zip file.

        Raises:
            ValueError: If API key is not set or zip file is invalid.
        """
        self.api_key = API_KEY
        if not self.api_key:
            raise ValueError("API_KEY is not set in the environment or .env file.")

        self.zip_file_path = zip_file_path
        if not os.path.exists(zip_file_path) or not zipfile.is_zipfile(zip_file_path):
            raise ValueError(f"Invalid zip file: {zip_file_path}")

        self.extracted_path = "extracted_repo"

    def extract_zip(self) -> None:
        """
        Extract the contents of the zip file to a temporary directory.
        """
        if os.path.exists(self.extracted_path):
            shutil.rmtree(self.extracted_path)

        os.makedirs(self.extracted_path)

        with zipfile.ZipFile(self.zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.extracted_path)

        logger.info(f"Repository extracted to {self.extracted_path}")

    def collect_code_files(self, root_dir: str) -> List[str]:
        """
        Traverse the extracted directory and collect code files.

        Args:
            root_dir (str): Root directory to start traversal.

        Returns:
            List[str]: List of code file contents.
        """
        code_list = []
        file_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)

                if (
                    os.path.getsize(file_path) > 10_000_000
                    or file.startswith(".")
                    or os.path.basename(root).startswith(".")
                ):
                    continue

                extension = file.split(".")[-1].lower()
                if extension in INCLUDE_EXTENSIONS:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            code_list.append(content)
                            file_paths.append(os.path.relpath(file_path, root_dir))
                    except (UnicodeDecodeError, FileNotFoundError) as e:
                        logger.warning(f"Could not read file {file_path}: {e}")

        logger.info(f"Collected {len(code_list)} code files")
        return code_list

    def run(self) -> str:
        """
        Run the README generation process.

        Returns:
            str: Generated README content.
        """
        try:
            self.extract_zip()
            code_list = self.collect_code_files(self.extracted_path)

            if not code_list:
                logger.warning("No valid code files found in the repository.")
                return "# README\n\nNo valid code files found in the repository."

            logger.info(f"Starting README generation with {len(code_list)} files")
            summarizer = LLMSummarize(self.api_key)
            readme_content = summarizer.summarize_repo(code_list)

            return readme_content

        except Exception as e:
            logger.error(f"Error generating README: {str(e)}")
            return f"# README Generation Error\n\nAn error occurred: {str(e)}"
        finally:
            if os.path.exists(self.extracted_path):
                shutil.rmtree(self.extracted_path)
                logger.info("Cleaned up extracted files")
