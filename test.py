# Import necessary modules
from llm_handler import LLMHandler

# Initialize LLMHandler
llm_handler = LLMHandler()

# Define a query
query = "What are a group of wolves called?"

# Generate a response
response = llm_handler.generate_response(query, [])

# Print the response
print("Response from LLM:")
print(response)