import json
import streamlit as st
from meta_ai_api import MetaAI

ai = MetaAI()

with open('dataset.json', 'r') as file:
    dataset = json.load(file)

questions_responses = [{"question": item["question"].lower(), "response": item["response"]} for item in dataset]

def find_response(user_query):
    user_words = set(user_query.lower().split())
    best_match = None
    max_overlap = 0

    for qa in questions_responses:
        question_words = set(qa["question"].split())
        overlap = len(user_words & question_words)
        
        if overlap > max_overlap:
            best_match = qa["response"]
            max_overlap = overlap

    return best_match

def get_meta_ai_response(user_query):
    dataset_string = json.dumps(dataset)
    prompt_message = f"String: {dataset_string}\nQuestion: {user_query}"
    response = ai.prompt(message=prompt_message)
    return response.get('message', "I'm sorry, I couldn't find an answer for that question.")

def log_question_and_response(question, response):
    with open("all_questions_answers.txt", "a") as log_file:
        log_file.write(f"Question: {question}\nResponse: {response}\n\n")

def chatbot():
    st.title("Chatbot Application")
    st.write("Chatbot is ready! Type your question below.")
    user_query = st.text_input("You:", "")

    if user_query:
        response = find_response(user_query)
        if response:
            st.write("Chatbot:", response)
        else:
            st.write("Chatbot: I'm consulting my knowledge base, please wait...")
            response = get_meta_ai_response(user_query)
            if response:
                st.write("Chatbot:", response)
            else:
                st.write("Chatbot: I'm sorry, I couldn't find an answer for that question.")
        log_question_and_response(user_query, response)

if __name__ == "__main__":
    chatbot()
