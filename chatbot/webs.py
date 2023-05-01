import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession
import json

def find_source(url):
    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def scrape(query, limit=5):
    link_list = "I do not understand...But I can still be of help with these links:  \n"
    query = urllib.parse.quote_plus(query)
    response = find_source("https://www.google.com/search?q=" + query)

    links = list(response.html.absolute_links)
    google_domains = ('https://www.google.', 
                      'https://google.', 
                      'https://webcache.googleusercontent.', 
                      'http://webcache.googleusercontent.', 
                      'https://policies.google.',
                      'https://support.google.',
                      'https://maps.google.')

    for url in links[:]:
        if url.startswith(google_domains):
            links.remove(url)
    
    for idx, link in enumerate(links[:limit]):
        link_list += "\t"+f"{idx + 1}. {link} "+"\n"

    return(link_list)
