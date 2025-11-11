# docusum-ai
Meringkas dokumen Tugas Akhir (TA)

https://docusum.vercel.app/

Jika ingin menjalankan pada terminal lokal:
- setx GOOGLE_API_KEY "AIzaSyAmwg5i1a3j5a1RkWtUvYmCLpqS6Fp58Qk"
- Install segala yang tertera di `\backend\requierements.txt` `pip install -r requirements.txt`
- Terminal 1 (Backend): `cd backend` lalu `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Terminal 2 (Frontend): `cd frontend` lalu `python -m http.server 5500`
- Buka di browser: http://localhost:5500/

Catatan: 
- Untuk komentar hanya berfungsi saat aplikasi sudah online, sesuai dengan `base_api/base URL`
