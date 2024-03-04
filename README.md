# Test LLM avec CPU

## Mise en route

Recupérer le modèle avec la commande `mc cp s3/tseimandi/llama-2-7b-chat.ggmlv3.q8_0.bin models/llama-2-7b-chat.ggmlv3.q8_0.bin` ou avec `mc cp s3/tseimandi/llama-2-7b-chat.ggmlv3.q4_K_M.bin models/llama-2-7b-chat.ggmlv3.q4_K_M.bin`. Ajuster le chemin vers le LLM en fonction dans `constants.py`. Dans ce même fichier, éditer la variable `S3_BUCKET_NAME` en inscrivant le nom de votre bucket personnel sur https://minio.lab.sspcloud.fr. Copier les quelques documents d'exemple sur MinIO avec la commande `mc cp -r data/ s3/<your-bucket>/test-rag/` (le nom de dossier correspond à la variable `S3_PDF_PREFIX` de `constants.py`).

Installer `llama-cpp-python` avec la commande bash `source install_llama_cublas.sh`, la librarie `unstructured` avec la command `source install_unstructured.sh` et les dépendances Python avec `pip install -r requirements.txt`. Puis construire le *vector store* avec `python src/db_build.py chroma` puis pour interroger le modèle, utiliser la commande `python src/main.py "What is the minimum guarantee payable by Adidas?"`.

## Application Streamlit

Localement, `streamlit run src/streamlit_app.py`.