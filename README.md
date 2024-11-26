# RAG Retrieval Techniques

This project demonstrates various retrieval techniques for document retrieval using Python. The techniques implemented include HyDe, Basic, Reciprocal Rank Fusion (RRF), and Fusion Retrieval. The project uses Streamlit for the user interface and various libraries for document processing and retrieval.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Retrieval Techniques](#retrieval-techniques)
  - [HyDe](#hyde)
  - [Basic](#basic)
  - [RRF](#rrf)
  - [Fusion](#fusion)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Upload a PDF file using the sidebar.

3. Select a retrieval technique from the sidebar.

4. Enter a query in the text input box and view the retrieved documents.

## Retrieval Techniques

### HyDe

HyDe (Hypothetical Document) retrieval generates a hypothetical document based on the query and retrieves similar documents.

### Basic

Basic retrieval uses a simple similarity search to retrieve documents based on the query.

### RRF

Reciprocal Rank Fusion (RRF) combines the results of multiple retrieval algorithms to improve the overall retrieval performance.

### Fusion

Fusion retrieval combines vector search and BM25 search results using a weighted sum to retrieve the most relevant documents.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.