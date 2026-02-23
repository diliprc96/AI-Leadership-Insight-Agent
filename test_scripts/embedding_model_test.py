import boto3
import json
import os
from dotenv import load_dotenv

# 1. LOAD THE ENV VARIABLES
load_dotenv() 

# Create a Bedrock Runtime client
# Boto3 will now automatically find the credentials in the environment
client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "amazon.titan-embed-text-v2:0"
input_text = "Please recommend books with a theme similar to the movie 'Inception'."

native_request = {"inputText": input_text}
request = json.dumps(native_request)

try:
    response = client.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())
    
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]

    print("\nYour input:")
    print(input_text)
    print(f"Number of input tokens: {input_token_count}")
    print(f"Size of the generated embedding: {len(embedding)}")
    # print("Embedding:", embedding) # Uncomment to see full vector

except Exception as e:
    print(f"Error: {e}")
