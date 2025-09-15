from eaa.gui.chat import set_message_db_path, run_webui

# Set your SQLite DB path here. It must match the task manager's message_db_path.
set_message_db_path("messages.db")

# Run the WebUI server
if __name__ == "__main__":
    run_webui(host="127.0.0.1", port=8008)
