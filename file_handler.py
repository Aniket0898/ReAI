import fitz
from llama_index.core import Document
import json, requests
import pandas as pd
import pytesseract, docx2txt
import pytesseract
import os
from PIL import Image

class FileHandler:
  """
  A class for downloading Google Drive files, extracting text, and creating Document objects.
  """
  def __init__(self, api_key):
    self.api_key = api_key

  def process_files(self, folder_url):
    """
    Downloads files from a Google Drive folder, extracts text, and returns a list of Document objects.
    Args:
        folder_url: The URL of the Google Drive folder.
    Returns:
        A list of Document objects.
    """

    downloaded_files = self.download_files(folder_url)
    documents = self.load_documents(downloaded_files)

    return documents

  def download_files(self, folder_url):
    """
    Downloads files from a Google Drive folder and stores them in a temporary directory.
    Args:
        folder_url: The URL of the Google Drive folder.
    Returns:
        A list of downloaded file paths.
    """
    folder_id = folder_url.split('/')[-1]
    endpoint = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}' in parents&key={self.api_key}"
    response = requests.get(endpoint)

    if response.status_code == 200:
      files = response.json().get('files', [])
      downloaded_files = []

      for file in files:
        file_name = file['name']
        file_id = file['id']
        download_url = f"https://drive.google.com/uc?id={file_id}"
        r = requests.get(download_url, allow_redirects=True)
        file_path = f'temp/{file_name}'  # Store in temp directory

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
          f.write(r.content)

        downloaded_files.append(file_path)

      return downloaded_files
    else:
      print(f"Failed to retrieve files from Google Drive. Status code: {response.status_code}")
      return []

  def load_documents(self, file_paths):
    """
    Loads documents from the given file paths and extracts text.
    Args:
        file_paths: A list of file paths.
    Returns:
        A list of Document objects.
    """
    documents = []
    for file_path in file_paths:
      try:
        text = ""
        if file_path.lower().endswith('.txt') or file_path.lower().endswith('.html'):
          with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        elif file_path.lower().endswith('.docx'):
          text = process(file_path)

        elif file_path.lower().endswith('.pdf'):
          with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
              page = doc.load_page(page_num)
              text += page.get_text()

        elif file_path.lower().endswith(('.xlsx', '.xls')):
          xls = pd.ExcelFile(file_path)
          sheet_names = xls.sheet_names
          for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text += df.to_string()

        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
          text = pytesseract.image_to_string(Image.open(file_path))

        else:
          print(f"Unsupported file format for '{file_path}'")

        documents.append(Document(text=text, metadata={"source": file_path}))

      except Exception as e:
        print(f"Error loading file '{file_path}': {e}")

    return documents
