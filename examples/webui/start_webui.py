from eaa_core.gui.nicegui import run_nicegui_webui

# Use the same SQLite file as the task manager's `session_db_path`.
# If you checkpoint the graph, point both the task manager and the WebUI at
# that same checkpoint database file.
SESSION_DB_PATH = "session.sqlite"

# Run the WebUI server
if __name__ == "__main__":
    run_nicegui_webui(SESSION_DB_PATH, host="127.0.0.1", port=8008)
