"""`python -m s2b_rag.web [--port 8000]`: chạy chatbot demo, mở http://127.0.0.1:8000."""

import argparse

import uvicorn

from s2b_rag.web.server import create_app, load_pipeline

parser = argparse.ArgumentParser(description="Chatbot demo S2B")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
uvicorn.run(create_app(load_pipeline), host=args.host, port=args.port)
