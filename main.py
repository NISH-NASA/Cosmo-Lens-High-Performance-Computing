"""
CosmoLens HPC: High-Performance JWST Deep-Field Processing & Gravitational Lens Discovery Engine
Primary entrypoint script.
"""

import sys
import os
import uvicorn

# Ensure repository root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"🚀 Launching CosmoLens HPC Observatory Server on http://localhost:{port}")
    uvicorn.run("cosmolens.server.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
