mkdir -p ~/.streamlit/
echo "[general]
email = \"zachgursky@yahoo.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = PORT
enableCORS = false
" > ~/.streamlit/config.toml