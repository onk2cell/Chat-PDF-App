import streamlit as st
from google.oauth2 import service_account
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2
from googleapiclient.discovery import build
import PyPDF2




# Authenticate with Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = r'claender-397805-882a0a3c233e.json'  # Update with your JSON file path

credentials = service_account.Credentials.from_service_account_file(
   SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

# This function will go through a PDF and extract and return a list of page texts.
import io

def read_and_textify(file_id):
    text_list = []
    sources_list = []

    request = drive_service.files().get_media(fileId=file_id)
    fh = request.execute()

    # Convert the bytes to a BytesIO object
    fh = io.BytesIO(fh)

    pdfReader = PyPDF2.PdfReader(fh)
    
    for i, pageObj in enumerate(pdfReader.pages):
        text = pageObj.extract_text()
        text_list.append(text)
        sources_list.append(f"Page {i + 1}")

    return text_list, sources_list
#This function will go through pdf and extract and return list of page texts which are being uploaded loclly.
def read_and_textify_from_local(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]
st.set_page_config(layout="centered", page_title="Multidoc_QnA")
st.header("Chat_with_pdf")
st.write("---")
  
#file uploader
uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
st.write("---")


#Path for your open ai api key 
path = st.text_input('Paste Your Open Ai API key Here')

if uploaded_files is None:
  st.info(f"""Upload files to analyse""")
elif uploaded_files:
  st.write(str(len(uploaded_files)) + " document(s) loaded..")
  
  textify_output = read_and_textify_from_local(uploaded_files)
  
  documents = textify_output[0]
  sources = textify_output[1]
  
  #extract embeddings
  embeddings = OpenAIEmbeddings(openai_api_key = path)
  #vstore with metadata. Here we will store page numbers.
  vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
  #deciding model
  model_name = "gpt-3.5-turbo"
  # model_name = "gpt-4"

  retriever = vStore.as_retriever()
  retriever.search_kwargs = {'k':2}

  #initiate model
  llm = OpenAI(model_name=model_name, openai_api_key = path, streaming=True)
  model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
  st.header("Ask your data")
  user_q = st.text_area("Enter your questions here")
  
  if st.button("Get Response"):
    try:
      with st.spinner("Model is working on it..."):
        result = model({"question":user_q}, return_only_outputs=True)
        st.subheader('Your response:')
        st.write(result['answer'])
        st.subheader('Source pages:')
        st.write(result['sources'])
    except Exception as e:
      st.error(f"An error occurred: {e}")
      st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')


st.title("Select PDF File From Google drive to chat with")

# Fetch a list of file names from Google Drive
results = drive_service.files().list().execute()
files = results.get('files', [])

# Extract file names from the list of files
list_of_file_names = [file['name'] for file in files]

# Create a file selector widget
selected_file_name = st.selectbox("Select a pdf file from Google Drive:", ["None"] + list_of_file_names)

if selected_file_name != "None":
    selected_file = [file for file in files if file['name'] == selected_file_name][0]
    file_id = selected_file['id']
    st.write(f"{selected_file_name} is selected.")

    # Call the read_and_textify function to extract text from the selected file
    textify_output = read_and_textify(file_id)
    documents = textify_output[0]
    sources = textify_output[1]

    #extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = path)
  #vstore with metadata. Here we will store page numbers.
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
  #deciding model
    model_name = "gpt-3.5-turbo"
  # model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k':2}

  #initiate model
    llm = OpenAI(model_name=model_name, openai_api_key = path, streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")
  
    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = model({"question":user_q}, return_only_outputs=True)
                st.subheader('Your response:')
                st.write(result['answer'])
                st.subheader('Source pages:')
                st.write(result['sources'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
      
    

    
