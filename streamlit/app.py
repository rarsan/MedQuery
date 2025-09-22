import streamlit as st
from google.cloud import bigquery, storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.vision_models import MultiModalEmbeddingModel
import pandas as pd
import base64
import re

# --- Configuration ---
PROJECT_ID = "bq-ai-compete"
REGION = "us-central1"
DATASET_ID = "pdf_analysis"
ARTICLES_TABLE = "articles"
PAGES_TABLE = "pages"
EMBEDDINGS_TABLE = "page_embeddings"
ARTICLE_LABELS_TABLE = "article_labels"

# --- Initialization ---
st.set_page_config(page_title="Biomedical Article Search", layout="wide")

@st.cache_resource
def init_clients():
    vertexai.init(project=PROJECT_ID, location=REGION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    storage_client = storage.Client(project=PROJECT_ID)
    return bq_client, storage_client

bq_client, storage_client = init_clients()

@st.cache_data
def get_image_bytes(_storage_client, gcs_uri):
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = _storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

@st.cache_data
def get_journals():
    query = f"""SELECT DISTINCT journal_book FROM `{PROJECT_ID}.{DATASET_ID}.{ARTICLES_TABLE}` ORDER BY journal_book"""
    query_job = bq_client.query(query)
    return [row["journal_book"] for row in query_job]

@st.cache_data
def get_study_types():
    query = f"""SELECT DISTINCT study_type FROM `{PROJECT_ID}.{DATASET_ID}.{ARTICLE_LABELS_TABLE}` WHERE study_type IS NOT NULL ORDER BY study_type"""
    query_job = bq_client.query(query)
    return [row["study_type"] for row in query_job]

@st.cache_data
def get_populations():
    query = f"""SELECT DISTINCT population FROM `{PROJECT_ID}.{DATASET_ID}.{ARTICLE_LABELS_TABLE}` WHERE population IS NOT NULL ORDER BY population"""
    query_job = bq_client.query(query)
    return [row["population"] for row in query_job]

@st.cache_data
def get_year_range():
    query = f"""SELECT MIN(publication_year) as min_year, MAX(publication_year) as max_year FROM `{PROJECT_ID}.{DATASET_ID}.{ARTICLES_TABLE}`"""
    query_job = bq_client.query(query)
    result = list(query_job)[0]
    return result["min_year"] or 2000, result["max_year"] or 2025

@st.cache_data
def get_embedding(text):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    embeddings = model.get_embeddings(contextual_text=text)
    return embeddings.text_embedding

# def get_total_counts(_bq_client, year_filter, journal_filter, study_type_filter, population_filter):
def get_total_counts(_bq_client, year_filter, journal_filter):
    journal_filter_str = ""
    if journal_filter:
        journal_filter_str = "AND a.journal_book IN (" + ", ".join([f"'{j}'" for j in journal_filter]) + ")"
    
    study_type_filter_str = ""
    # if study_type_filter:
    #     study_type_filter_str = "AND l.study_type IN (" + ", ".join([f"'{s}'" for s in study_type_filter]) + ")"

    population_filter_str = ""
    # if population_filter:
    #     population_filter_str = "AND l.population IN (" + ", ".join([f"'{p}'" for p in population_filter]) + ")"

    query = f"""
    WITH filtered_articles AS (
        SELECT a.pmcid
        FROM `{PROJECT_ID}.{DATASET_ID}.{ARTICLES_TABLE}` a
        WHERE a.publication_year BETWEEN {year_filter[0]} AND {year_filter[1]}
        {journal_filter_str}
        {study_type_filter_str}
        {population_filter_str}
    )
    SELECT 
        (SELECT COUNT(pmcid) FROM filtered_articles) as total_articles,
        (SELECT COUNT(p.page_id) 
         FROM `{PROJECT_ID}.{DATASET_ID}.{PAGES_TABLE}` p 
         WHERE p.pmcid IN (SELECT pmcid FROM filtered_articles)
        ) as total_pages
    """
    query_job = _bq_client.query(query)
    result = list(query_job)
    if not result:
        return 0, 0
    return result[0]["total_articles"], result[0]["total_pages"]

def perform_search(question, year_filter, journal_filter, limit):
    # question_embedding = get_embedding(question)

    journal_filter_str = ""
    if journal_filter:
        journal_filter_str = "AND journal_book IN (" + ", ".join([f"'{j}'" for j in journal_filter]) + ")"
    
    search_limit = limit
    
    query = f"""
     SELECT
        base.page_number, base.page_id, base.abstract, base.publication_year, base.journal_book, base.title, base.authors, base.page_image_uri, base.page_pdf_uri, distance
     FROM VECTOR_SEARCH(
        (SELECT * FROM {DATASET_ID}.{EMBEDDINGS_TABLE}
         WHERE publication_year BETWEEN {year_filter[0]} AND {year_filter[1]} {journal_filter_str}),
        'embedding',
        (SELECT ml_generate_embedding_result as embedding FROM ML.GENERATE_EMBEDDING(
            MODEL `{DATASET_ID}.mm_embedding_model`,
            (SELECT @question AS content)
        )),
        top_k => {search_limit},
        distance_type => 'COSINE',
        options => '{{\"fraction_lists_to_search\":0.15}}'
     )
     ORDER BY distance
     LIMIT {limit}"""
    
    # Basic escaping for SQL string literal
    escaped_question = question.replace("'", "\'")
    debug_query = query.replace("@question", f"'{escaped_question}'")

    # Store the debug query in session state for the sidebar to use
    st.session_state['debug_query'] = debug_query

    # Define the query job configuration with the named parameter for security.
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("question", "STRING", question),
        ]
    )

    query_job = bq_client.query(query, job_config=job_config)
    return list(query_job)

def generate_answer(_question, _search_results):
    model = GenerativeModel("gemini-2.5-pro")
    system_instruction = f"""You are a world-class biomedical research assistant. Your task is to answer the user's question based on the provided sources.

**Your response MUST follow these rules:**
1.  **Synthesize a Comprehensive Answer:** Create a detailed and easy-to-read answer from the text and images in the provided sources.
2.  **Provide Inline Citations:** When you use information from a source, you MUST cite it immediately using the format `[Source X]`.
3.  **Extract Key Data Points:** If you find specific, important data points (like percentages, statistics, measurements, p-values), present them clearly in a Markdown table.
4.  **Suggest Charts:** If you identify data that could be visualized (e.g., a comparison between groups, a trend over time), explicitly state that a chart would be useful and present the data for the chart in a clearly labeled Markdown table.

Here are the {len(_search_results)} sources to use:"""
    prompt_parts = [system_instruction]
    for i, row in enumerate(_search_results):
        prompt_parts.extend([f"---", f"**[Source {i + 1}]**", f"**Title:** {row['title']}", f"**Page:** {row['page_number']}", f"**Abstract:** {row['abstract']}", Part.from_uri(row['page_pdf_uri'], mime_type="application/pdf"), f"---"])
    prompt_parts.extend([f"\n\n**Question:** {_question}", f"\n\n**Answer:**"])
    response = model.generate_content(prompt_parts)
    return response.text

# --- UI ---
st.title("⚕️ Biomedical Library Research")

with st.sidebar:
    st.header("Filters")
    st.toggle("Debug Mode", key='debug_mode')
    st.divider()

    min_year, max_year = get_year_range()
    selected_years = st.slider("Publication Year", min_year, max_year, (min_year, max_year), key='selected_years') if min_year < max_year else (min_year, max_year)
    journals = get_journals()
    selected_journals = st.multiselect("Journals", journals, key='selected_journals')
    
    study_types = get_study_types()
    selected_study_types = st.multiselect("Study Types", study_types, key='selected_study_types')

    populations = get_populations()
    selected_populations = st.multiselect("Populations", populations, key='selected_populations')
    
    doc_options = {
        1: "I'm feeling lucky",
        5: "QA on narrow topic",
        10: "QA (most common)",
        25: "QA on broader topic",
        50: "Summarize top articles",
        100: "Summarize many articles (potentially slow and costly)",
    }
    option_values = list(doc_options.keys())
    def format_doc_option(value):
        return f"{value} - {doc_options[value]}"
    st.selectbox(
        "Number of documents to retrieve:", 
        options=option_values, 
        index=2, # Default to 10
        format_func=format_doc_option, 
        key='num_documents'
    )

    st.header("Corpus Stats")
    total_articles, total_pages = get_total_counts(bq_client, selected_years, selected_journals)
    c1, c2 = st.columns(2)
    c1.metric("Articles", f"{total_articles:,}")
    c2.metric("Pages", f"{total_pages:,}")

    if st.session_state.get('debug_mode', False):
        st.divider()
        st.header("Debug Info")

        # Display the last executed query if it exists in the session state
        if 'debug_query' in st.session_state:
            st.subheader("Vector Search Query")
            st.code(st.session_state['debug_query'], language="sql")

        st.subheader("Session State")
        st.write(st.session_state)

        if 'search_results' in st.session_state:
            st.subheader("Search Results (Debug)")
            search_results = st.session_state['search_results']
            debug_data = []
            for i, row in enumerate(search_results):
                debug_data.append({
                    "title": row.get('title'),
                    "page": row.get('page_number'),
                    "distance": row.get('distance'),
                    "page_image_uri": row.get('page_image_uri')
                })
            st.dataframe(pd.DataFrame(debug_data))

st.write("Ask a question about biomedical topics and get a synthesized answer from a corpus of articles.")
question = st.text_area("Ask your question here:", height=150, key="question_input")

if st.button("Search"):
    if st.session_state.question_input:
        with st.spinner("Performing semantic search and generating answer..."):
            search_results = perform_search(st.session_state.question_input, st.session_state.selected_years, st.session_state.selected_journals, st.session_state.num_documents)
            if search_results:
                st.session_state['search_results'] = search_results
                st.session_state['answer'] = generate_answer(st.session_state.question_input, search_results)
            else:
                st.warning("No relevant articles found for your query.")
                for key in ['search_results', 'answer']:
                    if key in st.session_state:
                        del st.session_state[key]
    else:
        st.warning("Please enter a question.")

# After the search controls, if an answer exists, create the two-column layout
if 'answer' in st.session_state:
    answer_col, detail_col = st.columns([0.6, 0.4])

    with answer_col:
        st.header("Answer")
        st.markdown(st.session_state['answer'])

    with detail_col:
        st.header("Sources")
        if 'search_results' in st.session_state:
            search_results = st.session_state['search_results']
            
            for i, row in enumerate(search_results):
                with st.expander(f"[{i + 1}] {row['title']} (Page {row['page_number']})"):
                    image_bytes = get_image_bytes(storage_client, row['page_image_uri'])
                    st.image(image_bytes, use_container_width=True)
                    st.write(f"**Journal:** {row['journal_book']}, **Year:** {row['publication_year']}")
                    st.write(f"**Authors:** {row['authors']}")
                    st.write(f"**Abstract:** {row['abstract']}")
        else:
            st.info("Search for a topic to see source details here.")

st.info("To run this app, you need to be authenticated with Google Cloud. Run `gcloud auth application-default login` in your terminal.")
