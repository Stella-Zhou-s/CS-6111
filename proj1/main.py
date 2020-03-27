#!/usr/bin/python3
import sys

import requests
import json

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
import numpy as np

global SEARCH_JSON_API_KEY, SEARCH_ENGINE_ID, TARGET_PRECISION, QUERY

def get_para():
    """
    Get parameters from command line
    """
    global SEARCH_JSON_API_KEY, SEARCH_ENGINE_ID, TARGET_PRECISION, QUERY

    SEARCH_JSON_API_KEY = input("Enter JSON_API_KEY：")
    SEARCH_ENGINE_ID = input("Enter SEARCH_ENGINE_ID: ")
    TARGET_PRECISION = float(input("Enter TARGET_PRECISION: "))
    QUERY = input("Enter Query: ")

    # Print to console
    print ("Search Key is : " + SEARCH_JSON_API_KEY)
    print ("Search Engine ID is : " + SEARCH_ENGINE_ID)
    print ("Target Precision is : " + str(TARGET_PRECISION))
    print ("QUERY is : " + QUERY)


def google_search():
    """
    Return Top-10 results of Google Custom Search API
    :return: search item list
    """
    url = "https://www.googleapis.com/customsearch/v1?q=" + QUERY \
          + "&cx=" + SEARCH_ENGINE_ID \
          + "&key=" + SEARCH_JSON_API_KEY

    response = requests.get(url)
    results = json.loads(response.text)['items']

    res = []
    for item in results:
        title = item['title']
        url = item['link']
        description = item['snippet']

        item_data = {
            "title": title,
            "url": url,
            "description": description,
            "relevant": False
        }
        res.append(item_data)

    return res


def get_user_feedback():
    """
    User Interface and get relevance information
    :return: search item list with relevance field
    """
    results = google_search()

    for idx in range(len(results)):
        item = results[idx]
        print ("Result " + str(idx + 1) + " is as following:")
        print (" Title: " + item['title'])
        print (" URL: " + item['url'])
        print (" Description: " + item['description'])

        feedback = input("Is This Document Relevant? (Y/N) ")
        if feedback == 'Y' or feedback == 'y':
            item['relevant'] = True
        print ("")

    return results


def get_precision(results):
    """
    Calculate the precision based on feedback
    :param results: list of results(10)
    :return: precision(float)
    """
    count = 0
    for item in results:
        if item['relevant']:
            count = count + 1

    return count * 1.0 / 10


def rocchio_relevance_feedback(query: str, ls_relev: list, ls_irrel: list, weights: tuple = None):
    """
    Using Rocchio algorithm to update query using relevance feedback.
    :param query: string of query
    :param ls_relev: list of relevant documents strings
    :param ls_irrel: list of irrelevant documents strings
    :param weights: (a,b,c) weights for query, relevant and irrelevant terms respectively.
    :return: list of tuples (word, weight)
    """
    text_collection = ls_relev + ls_irrel + [query]
    len_relev, len_irrel = len(ls_relev), len(ls_irrel)

    if weights is None:
        weights = (0.5, 0.35, 0.15)
    a, b, c = weights

    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'[a-zA-Z]{3,}')
    vectorizer.fit(text_collection)

    vec_query = vectorizer.transform([query]).todense()

    if len_relev == 0:
        term_relev = 0
    else:
        mat_relev = vectorizer.transform(ls_relev).todense()
        vec_relev = mat_relev.sum(axis=0)
        term_relev = vec_relev / len_relev

    if len_irrel == 0:
        term_irrel = 0
    else:
        mat_irrel = vectorizer.transform(ls_irrel).todense()
        vec_irrel = mat_irrel.sum(axis=0)
        term_irrel = vec_irrel / len_irrel

    vec_new = a * vec_query + b * term_relev - c * term_irrel

    tokens = vectorizer.get_feature_names()
    weights = np.asarray(vec_new).reshape(-1)

    return dict(zip(tokens, weights))


def get_body_text(url: str) -> str:
    """
    GET the content of a web page and retrieve its body text.
    :param url: link to target web page
    :return: list of words
    """
    content = requests.get(url)
    html = content.text
    soup = BeautifulSoup(html, "html.parser")
    return soup.body.get_txt()

def generate_new_words(results):
    """
    generate new query
    :param results: search item list [{'title', 'url', 'description', 'relevant'},...]
    :return: two new words
    """
    
    ls_relev, ls_irrel = get_doc_list(results)
    weights = (1, 0.75, 0.15)

    candidate_words = rocchio_relevance_feedback(QUERY, ls_relev, ls_irrel, weights)

    candidate_words_sorted = sorted(candidate_words, key=candidate_words.get, reverse=True)

    queries = QUERY.split(" ")
    new_words = []
    i = 0
    for w in candidate_words_sorted:
        if w in queries:
            continue

        new_words.append(w)
        i += 1
        if i == 2:
            break

    return new_words

def get_doc_list(results):
    """
    :param results: [{'title', 'url', 'description', 'relevant'},...]
    :return ls_relev: list of relevant documents strings
    :return ls_irrel: list of irrelevant documents strings
    """

    ls_relev = []
    ls_irrel = []

    for idx in range(len(results)):

        item = results[idx]

        title = item['title']
        description = item['description']

        if item["relevant"]:
            ls_relev.append(description)
        else:
            ls_irrel.append(description)

    return ls_relev, ls_irrel


def feedback(precision, new_words):
    """
    Print the update feedback to console
    :param precision: float, new_words: list
    :return: void
    """
    global QUERY
    print ("Current Query is: " + QUERY)
    print ("Current Precision is: " + str(precision))

    if precision < TARGET_PRECISION:
        print ("The Target Precision is: " + str(TARGET_PRECISION))

        words = ""
        for word in new_words:
            words = words + " " + word

        print ("Thr New Word are: " + words)
        QUERY += " " + words

    else:
        print ("We have reached the Target Precision: " + str(TARGET_PRECISION))

    print ("") # New line


def main():

    get_para()

    print ("Google Custom Search API Result:")
    print ("======================")
    print ("======================")

    results = get_user_feedback()

    precision = get_precision(results)
    global QUERY

    # Loop until condition satisfy
    while (precision > 0.0) and (precision < TARGET_PRECISION):

        print("Current Query is: " + QUERY)
        print("Current Precision is: " + str(precision))
        print("Target Precision is: " + str(TARGET_PRECISION))
        print("We have to generate New Words")
        print("")

        new_words = generate_new_words(results)

        words = ""
        for word in new_words:
            words = words + " " + word

        QUERY += " " + words
        print("The New Word are: " + words)
        print("New Query is: " + QUERY)
        print("")
        results = get_user_feedback()
        precision = get_precision(results)

    if((precision - 0.0 ) < 0.000001):
        print("We have reached the 0 Precision")
    else:
        print("Current Precision is: " + str(precision))
        print("We have reached the Target Precision: " + str(TARGET_PRECISION))

        
if __name__ == '__main__':
    main()