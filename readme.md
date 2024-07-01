# DOCUMENTATION

1. install vmware
2. install ollama
3. install the following:
Reference: https://stackoverflow.com/questions/73969269/error-could-not-build-wheels-for-hnswlib-which-is-required-to-install-pyprojec
!pip install --upgrade pip
!pip install -U langchain;
!pip install --include-deps langchain-core;
!pip install langchain-community;
!pip install langchain_experimental;
!pip install langchain-text-splitters;
!pip install langchain sentence-transformers;
!pip install langchainhub
!pip install gpt4all;
!pip install langchain-chroma;
!pip install unstructured;
4. clone down the repository


# REFERENCES
https://github.com/ollama/ollama/blob/main/docs/windows.md
https://medium.com/@danushidk507/ollama-ollama-in-windows-384899e054e4
https://medium.com/@shmilysyg/setup-rest-api-service-of-ai-by-using-local-llms-with-ollama-eb4b62c13b71
https://weaviate.io/blog/local-rag-with-ollama-and-weaviate
https://medium.com/@imabhi1216/implementing-rag-using-langchain-and-ollama-93bdf4a9027c

http://localhost:11434/
C:\Windows\System32\drivers\etc
netsh interface portproxy add v4tov4 listenaddress=192.168.1.17 listenport=11434 connectaddress=127.0.0.1 connectport=11434 

# CODE
