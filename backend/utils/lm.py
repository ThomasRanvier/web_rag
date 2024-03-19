import transformers
from transformers_stream_generator import init_stream_support
init_stream_support()
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


hf_folder = '/hf_home'


def get_llm_and_tokenizer(model_name='mistralai/Mistral-7B-Instruct-v0.1'):
    """Get the 4 bit quantized LLM and corresponding Tokenizer

    Args:
        model_name (str, optional): The LLM model to use (from HuggingFace). Defaults to 'mistralai/Mistral-7B-Instruct-v0.1'.

    Returns:
        tuple(LLM, Tokenizer): The LLM and the Tokenizer
    """
    ## Load the Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=hf_folder)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    ## Create the quantization configuration for the LLM
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=getattr(torch, 'float16'),
        bnb_4bit_use_double_quant=True,
    )

    ## Load the quantized LLM
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        cache_dir=hf_folder
    )
    
    return llm, tokenizer
    
    
def get_embedding_model(model_name='sentence-transformers/all-mpnet-base-v2'):
    """Get the embedding model used for the similiraty search

    Args:
        model_name (str, optional): The embedding model to use (from HuggingFace). Defaults to 'sentence-transformers/all-mpnet-base-v2'.

    Returns:
        EmbeddingModel: The embedding model
    """
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'}, cache_folder=hf_folder)


def get_mistral_prompt(system_prompt, query, context=''):
    """Formalize the given system prompt, user query and optional context as a Mistral prompt

    Args:
        system_prompt (str): The system prompt
        query (str): The user query
        context (str, optional): The retrieval context to use. Defaults to ''.

    Returns:
        str: The Mistral ready prompt
    """
    prompt = f'<s>[INST] {system_prompt} [/INST]</s>[INST] '
    if context != '':
        prompt += f'\n\n\nContext: {context}'
        prompt += f'\nQuestion: '
    prompt += f'{query} [/INST] '
    return prompt


def get_input_ids(prompt, tokenizer):
    """Get input ids from a Mistral prompt and its corresponding Tokenizer

    Args:
        prompt (str): The Mistral prompt
        tokenizer (Tokenizer): The tokenizer to use to encode the prompt

    Returns:
        Tensor: A tensor containing the list of tokens corresponding to the given prompt
    """
    input_ids = tokenizer(
        prompt, return_tensors='pt', add_special_tokens=False
    ).input_ids.to('cuda')
    return input_ids


def llm_query(prompt, llm, tokenizer, do_stream=False, temperature=.15, max_new_tokens=512, repetition_penalty=1.2):
    """Generate a response from the LLM, either as a generator for nice dynamic display or as a string for the standard output.

    Args:
        prompt (str): The prompt to feed to the LLM
        llm (transformers.ModelForCausalLM): The transfomers language model
        tokenizer (transformers.Tokenizer): The corresponding tokenizer
        do_stream (bool, optional): To return the response as a generator or as a string. Defaults to False.
        temperature (float, optional):  The value used to module the next token probabilities.
        max_new_tokens (int, optional): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        repetition_penalty (float, optional): The parameter for repetition penalty. 1.0 means no penalty.

    Returns:
        generator/str: The LLM response, either a generator or a string.
    """
    input_ids = get_input_ids(prompt, tokenizer)
    tokens = llm.generate(input_ids,
                          do_stream=do_stream,
                          do_sample=True,
                          temperature=temperature,
                          max_new_tokens=max_new_tokens,
                          repetition_penalty=repetition_penalty)
    if do_stream:
        return tokens
    else:
        return tokenizer.decode(tokens[0][len(input_ids[0]):], skip_special_tokens=True)


def reformulate_user_query(query, llm, tokenizer):
    """Reformulate the given query to better suit a web search

    Args:
        query (str): The user query
        llm (LLM): The Mistral LLM
        tokenizer (Tokenizer): The tokenizer

    Returns:
        str: The reformulated query
    """
    system_prompt = 'Give a succint rephrasing of the user question for a web browser search.'
    return llm_query(get_mistral_prompt(system_prompt, query), llm, tokenizer)


def rag_streaming_query(context, query, llm, tokenizer):
    """Queries an LLM using the RAG paradigm based on the given context and query

    Args:
        context (str): The retrieval context to use for the RAG query
        query (str): The user query
        llm (LLM): The LLM
        tokenizer (Tokenizer): The tokenizer

    Yields:
        str: Each word outputed by the LLM 
    """
    system_prompt = 'Elaborate a precise and concise answer to the Question below based on the web search information in the Context section, do not invent any fact and go straight to the point.'
    prompt = get_mistral_prompt(system_prompt, query, context=context)
    generator = llm_query(prompt, llm, tokenizer, do_stream=True)
    last_decoded_tokens = []
    for x in generator:
        tokens = x.cpu().numpy().tolist()
        word = tokenizer.decode(tokens, skip_special_tokens=True)
        if ' ' in tokenizer.decode(last_decoded_tokens + tokens, skip_special_tokens=True):
            word = ' ' + word
        last_decoded_tokens = tokens
        yield word


def split_text(text):
    """Splits the given text in chunks

    Args:
        text (str): The text to split

    Returns:
        list[docs]: Chunks
    """
    text_splitter = CharacterTextSplitter(
        separator=' ',
        chunk_size=400,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_text(text)


def create_vectorstore(docs, embedding_model):
    """Create a vector store based on texts and an embedding model

    Args:
        texts (list[str]): A list of the texts to embed
        embedding_model (EmbeddingModel): The embedding model

    Returns:
        FAISS: The vector store containing the embeddings
    """
    return FAISS.from_texts(texts=docs, embedding=embedding_model)