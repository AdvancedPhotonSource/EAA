from eaa_core.gui.html import run_html_webui

RUNTIME_URL = "http://127.0.0.1:8010"

if __name__ == "__main__":
    run_html_webui(RUNTIME_URL, host="127.0.0.1", port=8008)
