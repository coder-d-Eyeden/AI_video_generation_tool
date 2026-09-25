"""
Launcher for Bengal Scheme-to-Video AI Studio.
Starts FastAPI backend server with Uvicorn.
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  Bengal Scheme-to-Video AI Studio (Government of West Bengal)")
    print("  Starting server at http://127.0.0.1:8000")
    print("=" * 65)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
