# Biomedical Article Search Engine

This application is a web-based tool for researchers and medical professionals to perform semantic searches on a corpus of biomedical articles.

## Application Scope

The core features of this application include:

*   **Semantic Search:** Users can ask natural language questions about medical topics.
*   **Retrieval-Augmented Generation (RAG):** The application uses a RAG model to retrieve relevant article excerpts and generate a synthesized answer to the user's query.
*   **Filtering:** Users can filter the search results by:
    *   **Year of Publication:** A slider to select a range of years.
    *   **Number of Documents:** A slider to control the number of documents retrieved.
    *   **Journal:** A multi-select dropdown to choose from a list of available journals.
*   **Display:** The application will display the synthesized answer, along with the source articles, including titles, authors, and links to the original articles.

## Architecture and Deployment Recommendation

This application is built as a **Streamlit application**.

### Why Streamlit?

*   **Speed of Implementation:** Streamlit is a Python-based framework that allows for rapid development of data-centric applications. Since the existing RAG logic is in Python, we can quickly integrate it into a Streamlit app.
*   **Visually Appealing:** Streamlit provides a clean and modern UI out-of-the-box, which aligns with the hackathon's goal of creating a visually appealing application.
*   **Contained Environment:** A Streamlit app can be a single Python script, making it easy to manage and share.
*   **Deployment:** For the hackathon, we can run the Streamlit app locally. If needed, it can be easily containerized and deployed to Google Cloud Run for a scalable, serverless solution.

