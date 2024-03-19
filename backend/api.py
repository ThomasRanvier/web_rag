from utils.lm import *
from utils.web_search import *

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import logging
import os


ROOT_LEVEL = os.environ.get('PROD', 'INFO')

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'},
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'level': ROOT_LEVEL,
            'handlers': ['default'],
            'propagate': False,
        },
        'uvicorn.error': {
            'level': 'DEBUG',
            'handlers': ['default'],
        },
        'uvicorn.access': {
            'level': 'DEBUG',
            'handlers': ['default'],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


logger = logging.getLogger(__name__)
app = FastAPI()


class Query(BaseModel):
    content: str


## Initiate models
logger.info(f'Torch: {torch.__version__}')
logger.info(f'Cuda available: {torch.cuda.is_available()}')
logger.info('Loading LLM and Tokenizer')
llm, tokenizer = get_llm_and_tokenizer()
logger.info('Loading Embedding Model')
embedding_model = get_embedding_model()
logger.info('Done')


def web_search_llm_api(query):
    """Search relevant information from the web and query the LLM using the RAG paradigm

    Args:
        query (str): The query to ask the LLM
        retrieve_k (int): How many documents to retrieve for RAG

    Yields:
        str: Each intermediary step as a line, then each word yielded from the LLM
    """
    ## Reformulate user query for web search
    # reformulated_query = reformulate_user_query(query, llm, tokenizer)
    reformulated_query = query # No reformulation as Mistral 7B sucks
    ## Get top 4 search results on reformulated query
    yield f'- Searching web\n'
    logger.info('- Searching web')
    search_results = web_search(reformulated_query, top_k=4)
    ## Prepare list of chunks
    docs = []
    for url in search_results:
        ## Scrap text from URL
        yield f'-     Sraping webpage: {url}\n'
        logger.info(f'-     Sraping webpage: {url}')
        scraped_text = scrap_webpage(url)
        yield f'-     Processing extracted text\n'
        logger.info('-     Processing extracted text')
        ## Split text in chunks
        chunks = split_text(scraped_text)
        ## Add source to each chunk
        url_docs = [f'Information: {chunk}\nSource: {url}\n\n' for chunk in chunks]
        docs += url_docs
    ## Create vector db with docs (with metadata to cite source)
    yield f'- Creating vector database from scrapped data\n'
    logger.info('- Creating vector database from scrapped data')
    vectorstore = create_vectorstore(docs, embedding_model)
    ## Similarity search to extract top 5 results from vector db
    yield f'- Querying most similar search hits\n'
    logger.info('- Querying most similar search hits')
    docs = vectorstore.similarity_search(query, k=5)
    context = ''
    for doc in docs:
        context += doc.dict()['page_content']
    yield f'- Querying LLM with web scrapped context\n'
    logger.info('- Querying LLM with web scrapped context')
    yield f'Answer:\n'
    for word in rag_streaming_query(context, query, llm, tokenizer):
        yield word


@app.post('/search')
async def search(query: Query):
    """Queries the LLM using the RAG paradigm

    Args:
        query (Query): Contains the text of the user query

    Returns:
        StreamingResponse: The streaming output of the LLM to answer the query
    """
    return StreamingResponse(web_search_llm_api(query.content), media_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)