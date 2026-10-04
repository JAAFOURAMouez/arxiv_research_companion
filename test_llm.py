#!/usr/bin/env python3
"""
Test script to check if the Hugging Face Inference API is working
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

def test_hf_inference():
    """Test the Hugging Face Inference API directly"""
    print("Testing Hugging Face Inference API...")

    # Get configuration from environment
    model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    api_token = os.getenv("HF_API_TOKEN", "")
    provider = os.getenv("HF_INFERENCE_PROVIDER", None)

    print(f"Model: {model_name}")
    print(f"Token set: {bool(api_token)}")
    print(f"Provider: {provider}")

    if not api_token:
        print("ERROR: HF_API_TOKEN not set in environment")
        return False

    try:
        # Initialize client
        client_options = {'model': model_name, 'token': api_token}
        if provider:
            client_options['provider'] = provider

        print(f"Initializing client with options: {client_options}")
        client = InferenceClient(**client_options)

        # Test with a simple prompt
        test_prompt = "What is the capital of France?"
        print(f"Sending test prompt: {test_prompt}")

        response = client.chat_completion(
            messages=[{'role': 'user', 'content': test_prompt}],
            max_tokens=50,
            temperature=0.7
        )

        print(f"Raw response: {response}")

        # Extract content
        choices = response.get('choices') if isinstance(response, dict) else getattr(response, 'choices', None)
        if choices:
            message = choices[0].get('message') if isinstance(choices[0], dict) else getattr(choices[0], 'message', None)
            content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)
            print(f"Response content: {content}")
            return True
        else:
            print("ERROR: No choices in response")
            return False

    except Exception as e:
        print(f"ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hf_inference()
    if success:
        print("\n✅ LLM test PASSED")
    else:
        print("\n❌ LLM test FAILED")