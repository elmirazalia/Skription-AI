gemma2:9b

Jika ingin menjalankan pada terminal lokal:
- Install segala yang tertera di `\backend\requierements.txt` `pip install -r requirements.txt`
- Terminal 1 (Backend): `cd backend` lalu `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Terminal 2 (Frontend): `cd frontend` lalu `python -m http.server 5500`
- Buka di browser: http://localhost:5500/
