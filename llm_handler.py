from dotenv import load_dotenv
import os
import json
import openai
from typing import List, Dict, Any
from datetime import datetime

dotenv_path = "environment/keys.env"
load_dotenv(dotenv_path=dotenv_path)

MODEL="gpt-4o-mini"
OPEN_API_KEY= os.getenv("OPEN_AI_API_KEY_2")

class LLMHandler:
    def __init__(self, api_key=None, model=MODEL):
        self.api_key = api_key or OPEN_API_KEY
        if not self.api_key:
            raise ValueError("API key is required. Set up an Open API key to USE")

        openai.api_key = self.api_key
        self.model = model
        self.conversations_dir = "conversations"
        os.makedirs(self.conversations_dir, exist_ok=True)

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a response using the LLM with retrieved context"""
        # Build the prompt with context
        system_prompt = "You are a helpful assistant that answers questions based on the provided context. "
        system_prompt += "If the answer cannot be found in the context, say so clearly."

        # Format context
        context_text = "\n\n".join([
            f"--- Document: {chunk['metadata']['document_name']}, Page: {chunk['metadata']['page']} ---\n{chunk['content']}"
            for chunk in context_chunks
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]

        try:
            # Call the OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def save_conversation(self, topic: str, messages: List[Dict[str, str]]) -> str:
        """Save a conversation to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{topic.replace(' ', '_')}_{timestamp}.json"
        filepath = os.path.join(self.conversations_dir, filename)

        with open(filepath, 'w') as f:
            json.dump({
                "topic": topic,
                "timestamp": timestamp,
                "messages": messages
            }, f, indent=2)

        return filepath

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations"""
        conversations = []
        for filename in os.listdir(self.conversations_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.conversations_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    conversations.append({
                        "filename": filename,
                        "topic": data.get("topic", "Untitled"),
                        "timestamp": data.get("timestamp", ""),
                        "message_count": len(data.get("messages", []))
                    })

        return sorted(conversations, key=lambda x: x["timestamp"], reverse=True)

    def load_conversation(self, filename: str) -> Dict[str, Any]:
        """Load a conversation from disk"""
        filepath = os.path.join(self.conversations_dir, filename)
        if not os.path.exists(filepath):
            return {"error": "Conversation not found"}

        with open(filepath, 'r') as f:
            return json.load(f)


# Example usage
if __name__ == "__main__":
    llm = LLMHandler()

    # Example context chunks
    context = [
        {
            "content": "The architecture follows a microservice pattern with separate services for user management and data processing.",
            "metadata": {"document_name": "architecture.pdf", "page": 1}
        }
    ]

    response = llm.generate_response("What architecture does the system use?", context)
    print(response)

    # Save conversation
    conversation = [
        {"role": "user", "content": "What architecture does the system use?"},
        {"role": "assistant", "content": response}
    ]
    llm.save_conversation("Architecture Discussion", conversation)