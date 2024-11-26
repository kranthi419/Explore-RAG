# RAG Retrieval Techniques

This project demonstrates the use of Retrieval-Augmented Generation (RAG) techniques for document retrievals. The project is implemented in Python and uses Streamlit for the web interface.

## Requirements

- `Python 3.11`
- `langchain-openai`
- `langchain-core`
- `python-dotenv`
- `streamlit`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a PDF file and enter your query to retrieve similar documents using the HyDe technique.

## Author

Kavali Kranthi Kumar

## License

This project is licensed under the MIT License.