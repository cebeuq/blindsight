import logging

import server

# Quiet test output: SDK per-request INFO chatter and the expected ask_image failure warnings.
server.setup_logging("WARNING")
logging.getLogger("vision-mcp").setLevel(logging.ERROR)
