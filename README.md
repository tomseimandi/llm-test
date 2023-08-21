# Test LLM avec CPU

## Mise en route

Recupérer le modèle avec la commande `mc cp s3/tseimandi/llama-2-7b-chat.ggmlv3.q8_0.bin models/llama-2-7b-chat.ggmlv3.q8_0.bin`.

Installer `llama-cpp-python` avec la commande bash `./install_llama_cublas.sh`. Puis construire le *vector store* avec `python src/db_build.py` puis pour interroger le modèle, utiliser la commande `python src/main.py "What is the minimum guarantee payable by Adidas?"`.

## Application Streamlit

Localement, `streamlit run src/streamlit_app.py`.