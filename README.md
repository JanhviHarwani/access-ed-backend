# Access-Ed-Assistant
to start scraping:
cd backend
source venv/bin/activate
cd src/scraping_scripts 
python web_processor.py

start backend:
source venv/bin/activate
pip install -r requirements.txt
cd src
python main.py

An intelligent chatbot that assists faculty in creating inclusive educational environments. Provides real-time guidance on accessibility accommodations, course material adaptation, and inclusive teaching strategies to support students with diverse learning needs.

accessibility_bot/
│
├── data/                    # Storing accessibility documents here
│   └── documents.txt
│
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # Document processing logic
│   ├── embeddings_manager.py    # Embeddings and vector store logic
│   ├── chat_interface.py        # Gradio interface
│   └── main.py                  # Main application file
|
├── .env                     # Environment variables
└── requirements.txt         # Project dependencies

commands:
source venv/bin/activate
pip install -r requirements.txt
cd src
python main.py


Last update:
process processed folder
need to scrape off of urls.txt has content for research on Research on accessing CS
- to add data into pinecone document_processor and pinecone manager has some code differences in the new github repo, hence use the older repo for uploading data into pinecone vector db

