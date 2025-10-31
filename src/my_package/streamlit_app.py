import os
import time
from typing import Optional

import streamlit as st
import openai
from dotenv import load_dotenv

load_dotenv()


def configure_openai(provider: str, api_key: Optional[str], endpoint: Optional[str], deployment: Optional[str]):
    """Configure the openai client for either 'openai' or 'azure'.

    For Azure OpenAI we set api_type, api_base and api_version and use the deployment name when calling chat completions.
    """
    if provider == "azure":
        if not api_key or not endpoint or not deployment:
            raise ValueError("Azure configuration requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
        openai.api_type = "azure"
        openai.api_base = endpoint
        # Use a common supported version; can be overridden via env if needed
        openai.api_version = "2023-05-15"
        openai.api_key = api_key
        return {"deployment": deployment}

    # default: OpenAI
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
    openai.api_type = "open_ai"
    openai.api_key = api_key
    return {"model": deployment or "gpt-3.5-turbo"}


def stream_completion(call_args):
    """Generator wrapper for streaming chat completions from openai.

    Yields text chunks as they arrive.
    """
    for chunk in openai.ChatCompletion.create(stream=True, **call_args):
        # payload parsing depends on the library; each chunk may contain choices
        if "choices" in chunk:
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    yield text


def main():
    st.set_page_config(page_title="Cliniq — OpenAI / Azure OpenAI Demo", layout="wide")
    st.title("Cliniq — Streamlit interface for OpenAI & Azure OpenAI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Settings")
        provider = st.selectbox("Provider", ["openai", "azure"], index=0)

        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
            endpoint = None
            deployment = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        else:
            api_key = st.text_input("Azure OpenAI Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
            endpoint = st.text_input("Azure Endpoint (e.g. https://<your-resource>.openai.azure.com)", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
            deployment = st.text_input("Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))

        temperature = st.slider("Temperature", 0.0, 1.0, 0.5)

    with col2:
        st.header("Chat")
        prompt = st.text_area("Prompt", height=200)

        run = st.button("Send")

        if run:
            try:
                config = configure_openai(provider, api_key, endpoint, deployment)
            except ValueError as e:
                st.error(str(e))
                return

            call_args = {
                **config,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
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
