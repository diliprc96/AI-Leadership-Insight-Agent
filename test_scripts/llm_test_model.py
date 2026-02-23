import boto3
import os
import json
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# 1. Load credentials from your .env
load_dotenv()

# 2. Create session using YOUR specific environment variables
# This ensures we use the "second key" you mentioned earlier.
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

# 3. Use the session to create the client
client = session.client("bedrock-runtime")

# 4. Set the model ID (Nova Pro is best for your balanced prototype)
model_id = "amazon.nova-pro-v1:0"

# 5. Define the user message
user_message = "Confirm you are Nova Pro and describe your primary advantage for AI agents."

conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    print(f"Talking to {model_id} via Converse API...")
    
    # The Converse API is simpler: it handles the JSON wrapping for you
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={
            "maxTokens": 1000, 
            "temperature": 0.3, # Lower temperature is better for agent consistency
            "topP": 0.9
        },
    )

    # Extract the response text
    # Note: 'content' is a list, so we take index [0]
    response_text = response["output"]["message"]["content"][0]["text"]
    
    print("\n--- Response ---")
    print(response_text)

    # Print usage for cost tracking
    usage = response.get("usage", {})
    print(f"\n[Tokens Used: In={usage.get('inputTokens')}, Out={usage.get('outputTokens')}]")

except ClientError as e:
    print(f"AWS ERROR: {e.response['Error']['Message']}")
except Exception as e:
    print(f"GENERAL ERROR: {e}")



# import boto3
# import json
# import os
# from dotenv import load_dotenv
# from botocore.exceptions import ClientError

# load_dotenv()

# # Use us-east-1 as the base region
# client = boto3.client("bedrock-runtime", region_name="us-east-1")

# # Use the US Inference Profile ID you confirmed exists
# model_id = "us.anthropic.claude-sonnet-4-6"

# native_request = {
#     "anthropic_version": "bedrock-2023-05-31",
#     "max_tokens": 512,
#     "temperature": 0.5,
#     "messages": [
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": "Are you Claude 4.6? Confirm your model version."}],
#         }
#     ],
# }

# try:
#     print(f"Attempting to invoke: {model_id}...")
#     response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
    
#     model_response = json.loads(response["body"].read())
#     # Note: Modern Claude responses usually nest text in ['content'][0]['text']
#     print("\nSUCCESS!")
#     print(f"Response: {model_response['content'][0]['text']}")

# except ClientError as e:
#     code = e.response['Error']['Code']
#     if code == 'AccessDeniedException':
#         print(f"\n❌ ACCESS DENIED: Your IAM User/Role needs 'bedrock:InvokeModel' on the profile resource.")
#         print(f"Target Resource: arn:aws:bedrock:us-east-1:*:inference-profile/{model_id}")
#     else:
#         print(f"❌ AWS Error: {e}")
# except Exception as e:
#     print(f"❌ Script Error: {e}")

# import boto3
# from dotenv import load_dotenv
# load_dotenv()
# # Initialize the Bedrock control plane client
# bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')

# def get_available_models():
#     try:
#         # Fetch all foundation models
#         response = bedrock.list_foundation_models()
#         models = response.get('modelSummaries', [])
        
#         print(f"{'Provider':<15} | {'Model ID':<45}")
#         print("-" * 65)
        
#         # Filter for ACTIVE models only
#         for model in models:
#             if model.get('modelLifecycle', {}).get('status') == 'ACTIVE':
#                 print(f"{model['providerName']:<15} | {model['modelId']:<45}")
                
#     except Exception as e:
#         print(f"Error fetching models: {e}")

# if __name__ == "__main__":
#     get_available_models()
