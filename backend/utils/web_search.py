from websearch import WebSearch as web
import bs4, requests


def scrap_webpage(url):
    """Scrap the given webpage

    Args:
        url (str): The URL of the webpage to scrap

    Returns:
        str: Raw extracted text from the webpage
    """
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    return soup.body.get_text(' ', strip=True)


def web_search(query, top_k=4):
    """Outputs the top k URLs that match the given query

    Args:
        query (str): The user query
        top_k (int, optional): The top k URLs to return. Defaults to 4.

    Returns:
        list[str]: List of the top k URLs
    """
    search_results = []
    clean_results = []
    for page in web(query).pages:
        clean_page = page[page.index('//')+2:]
        if clean_page not in clean_results:
            search_results.append(page)
            clean_results.append(clean_page)
        if len(search_results) >= top_k:
            break
    return search_results