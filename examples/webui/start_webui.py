from eaa.gui.chat import set_message_db_path, run_webui

# Use the same SQLite file as the task manager's `session_db_path`.
# If you checkpoint the graph, point both the task manager and the WebUI at
# that same checkpoint database file.
set_message_db_path("session.sqlite")

# Run the WebUI server
if __name__ == "__main__":
    run_webui(host="127.0.0.1", port=8008)
