import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from pathlib import Path

@dataclass
class QAPair:
    question: str
    response: str = ""

class DataManager:
    def __init__(self, dataset_path: str, unanswered_path: str):
        script_dir = Path(__file__).parent.resolve()
        self.dataset_path = script_dir / dataset_path
        self.unanswered_path = script_dir / unanswered_path
        self._initialize_files()
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Download required NLTK data"""
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def _initialize_files(self):
        """Ensure data files exist"""
        for file_path in [self.dataset_path, self.unanswered_path]:
            if not file_path.exists():
                file_path.write_text('[]')
                
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load data from JSON file"""
        try:
            data = json.loads(file_path.read_text())
            return data
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return []
            
    def _save_json(self, data: List[Dict], file_path: Path) -> bool:
        """Save data to JSON file"""
        try:
            file_path.write_text(json.dumps(data, indent=4))
            return True
        except Exception as e:
            st.error(f"Error saving to {file_path}: {e}")
            return False
            
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for matching"""
        tokens = word_tokenize(text.lower())
        return [
            self.stemmer.stem(word) 
            for word in tokens 
            if word.isalnum() and word not in self.stop_words
        ]
        
    def find_best_response(self, query: str) -> Optional[str]:
        """Find best matching response for query"""
        dataset = self._load_json(self.dataset_path)
        if not dataset:
            return None
            
        query_tokens = set(self.preprocess_text(query))
        best_match = None
        highest_score = 0
        
        for qa in dataset:
            qa_tokens = set(self.preprocess_text(qa['question']))
            # Calculate Jaccard similarity
            intersection = len(query_tokens & qa_tokens)
            union = len(query_tokens | qa_tokens)
            score = intersection / union if union > 0 else 0
            
            if score > highest_score:
                highest_score = score
                best_match = qa['response']
        
        # Only return if we have a reasonable match
        return best_match if highest_score > 0.3 else None
        
    def add_unanswered(self, question: str) -> bool:
        """Add question to unanswered list"""
        unanswered = self._load_json(self.unanswered_path)
        
        # Check if question already exists
        if not any(qa['question'] == question for qa in unanswered):
            unanswered.append({'question': question, 'response': ''})
            return self._save_json(unanswered, self.unanswered_path)
        return True
        
    def get_unanswered(self) -> List[QAPair]:
        """Get list of unanswered questions"""
        data = self._load_json(self.unanswered_path)
        return [QAPair(**qa) for qa in data]
        
    def save_answer(self, question: str, answer: str) -> bool:
        """Save answer to dataset and remove from unanswered"""
        # Add to dataset
        dataset = self._load_json(self.dataset_path)
        dataset.append({'question': question, 'response': answer})
        if not self._save_json(dataset, self.dataset_path):
            return False
            
        # Remove from unanswered
        unanswered = self._load_json(self.unanswered_path)
        unanswered = [qa for qa in unanswered if qa['question'] != question]
        return self._save_json(unanswered, self.unanswered_path)

class ChatbotUI:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("Educational Chatbot")
        st.write("Ask a question or help improve the chatbot by answering unanswered questions.")
        
        # Initialize session state for messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Chat input
        if query := st.chat_input("Ask your question"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Get response
            response = self.data_manager.find_best_response(query)
            
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I don't have an answer for that question yet. I've saved it for our experts to answer later."
                })
                self.data_manager.add_unanswered(query)
                
        # Help section
        if st.button("Help Improve Chatbot"):
            self.render_help_section()
            
    def render_help_section(self):
        """Render the section for answering unanswered questions"""
        st.write("### Help Improve the Chatbot")
        st.write("Below are questions that need answers. Your contributions help make the chatbot smarter!")
        
        unanswered = self.data_manager.get_unanswered()
        if not unanswered:
            st.write("No unanswered questions at the moment!")
            return
            
        for i, qa in enumerate(unanswered):
            with st.expander(f"Question: {qa.question}", expanded=True):
                answer = st.text_area(
                    "Your answer:",
                    key=f"answer_{i}",
                    help="Provide a clear and helpful answer"
                )
                
                if st.button("Submit", key=f"submit_{i}"):
                    if answer.strip():
                        if self.data_manager.save_answer(qa.question, answer.strip()):
                            st.success("Answer saved successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to save answer. Please try again.")
                    else:
                        st.warning("Please provide an answer before submitting.")

def main():
    # Initialize data manager
    data_manager = DataManager(
        dataset_path="dataset.json",
        unanswered_path="unanswered_questions.json"
    )
    
    # Initialize and run UI
    chatbot = ChatbotUI(data_manager)
    chatbot.render_chat_interface()

if __name__ == "__main__":
    main()
