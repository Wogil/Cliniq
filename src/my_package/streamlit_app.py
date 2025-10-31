import os
import time
from typing import Optional

import streamlit as st
import openai
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()


def configure_openai():
    """Configure the openai client for Azure OpenAI using environment variables.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")

    if not endpoint or not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required")

    # Parse the full endpoint URL to extract deployment and api version
    parsed = urlparse(endpoint)
    path = parsed.path or ""
    
    # Extract deployment name from path
    if "/deployments/" not in path:
        raise ValueError("AZURE_OPENAI_ENDPOINT must contain '/deployments/<name>/'")
    
    parts = [p for p in path.split("/") if p]
    try:
        dep_idx = parts.index("deployments")
        deployment_name = parts[dep_idx + 1]
    except (ValueError, IndexError):
        raise ValueError("Could not parse deployment name from AZURE_OPENAI_ENDPOINT")

    # Extract API version from query params if present
    if parsed.query:
        params = parse_qs(parsed.query)
        if "api-version" in params:
            api_version = params["api-version"][0]

    # Build api_base without the deployment path and query
    api_base = f"{parsed.scheme}://{parsed.netloc}"
    
    # Configure OpenAI client
    openai.api_type = "azure"
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key
    
    return {"engine": deployment_name}  # Use engine instead of deployment for Azure


def stream_completion(call_args):
    """Generator wrapper for streaming chat completions from openai.

    Yields text chunks as they arrive.
    """
    # Engine is already properly set in call_args from configure_openai()
    for chunk in openai.ChatCompletion.create(
        stream=True,
        **call_args
    ):
        # payload parsing depends on the library; each chunk may contain choices
        if "choices" in chunk:
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    yield text


def main():
    st.set_page_config(page_title="Cliniq — Chat Interface", layout="wide")
    st.title("Cliniq — Chat Interface")

    # Chat interface
    st.header("Chat")
    prompt = st.text_area("Prompt", height=200)
    run = st.button("Send")

    if run:
        try:
            config = configure_openai()
        except ValueError as e:
            st.error(str(e))
            return

        call_args = {
            **config,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,  # Fixed temperature value
        }

        placeholder = st.empty()
        full_text = ""

        with placeholder.container():
            st.markdown("**Assistant:**")
            message_area = st.empty()

        try:
            for chunk in stream_completion(call_args):
                full_text += chunk
                # Throttle updates slightly for nicer UI
                message_area.markdown(full_text)
                time.sleep(0.02)
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
