
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def validate_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only valid HTTP/HTTPS URLs are supported.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    host = parsed.hostname

    if host.lower() == "localhost" or host.endswith(".localhost"):
        raise ValueError("Local URLs are not allowed.")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, None)
        }
        if not addresses or any(
            not ipaddress.ip_address(ip).is_global
            for ip in addresses
        ):
            raise ValueError("Private or non-public hosts are not allowed.")
    except (socket.gaierror, ValueError) as exc:
        raise ValueError(f"Unsafe or unreachable host: {host}") from exc

    return url


def fetch_webpage(url, max_redirects=5):
    session = requests.Session()
    session.trust_env = False

    try:
        for _ in range(max_redirects + 1):
            validate_url(url)

            response = session.get(
                url,
                timeout=(5, 15),
                allow_redirects=False,
                headers={"User-Agent": "ProductionRAG/1.0"},
            )

            if response.is_redirect:
                location = response.headers.get("Location")
                response.close()

                if not location:
                    raise ValueError("Redirect has no destination.")

                url = urljoin(url, location)
                continue

            response.raise_for_status()

            if "text/html" not in response.headers.get(
                "Content-Type", ""
            ).lower():
                response.close()
                raise ValueError("URL must return an HTML page.")

            soup = BeautifulSoup(response.content, "html.parser")
            final_url = response.url
            response.close()
            break
        else:
            raise ValueError("Too many redirects.")

    except requests.RequestException as exc:
        raise ValueError(f"Could not fetch URL: {exc}") from exc
    finally:
        session.close()

    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "iframe", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else final_url
    text = soup.get_text(" ", strip=True)

    if len(text) < 50:
        raise ValueError("Page contains too little readable text.")

    return title, text, final_url


def load_and_split_url(url):
    title, text, final_url = fetch_webpage(url)

    document = Document(
        page_content=text,
        metadata={
            "source_type": "url",
            "source": final_url,
            "title": title,
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents([document])

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"url_{index}"

    return chunks
