# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from collections import Counter
import re, os, string, math, asyncio, time, json, requests
from typing import List, Dict, Any
from datetime import datetime

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

# CONFIG & PARAMETER
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")

MAX_CONCURRENCY = 10
MAX_RETRIES = 4
RETRY_BASE_DELAY = 0.8
OLLAMA_TIMEOUT = 180
MAX_INPUT_CHARS = 50000

# PDF TEXT EXTRACTION
def read_pdf_text(path: str) -> str:
    text = ""
    try:
        import fitz
        doc = fitz.open(path)
        text = "\n".join([p.get_text() for p in doc])
        text = clean_text(text)
        if _enough_text(text):
            return text
    except:
        pass

    try:
        from pdfminer.high_level import extract_text
        text = clean_text(extract_text(path))
        if _enough_text(text):
            return text
    except:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        raw = "\n".join([p.extract_text() or "" for p in reader.pages])
        text = clean_text(raw)
        if _enough_text(text):
            return text
    except:
        pass

    try:
        from pdf2image import convert_from_path
        import pytesseract
        pages = convert_from_path(path, dpi=300)
        ocr_text = ""
        for pg in pages:
            ocr_text += pytesseract.image_to_string(pg, lang="eng+ind") + "\n"
        return clean_text(ocr_text)
    except:
        return clean_text(text)

def _enough_text(text, min_chars=200):
    return len(text.strip()) >= min_chars

BLACKBOX = ["■","□","▯","█","�"]
def clean_text(text):
    if not text:
        return ""
    for b in BLACKBOX:
        text = text.replace(b, "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Gambar\s*\d+(\.\d+)*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Tabel\s*\d+(\.\d+)*", "", text, flags=re.IGNORECASE)
    return text.strip()

def clean_reference_noise(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\([A-Za-z][^()]{0,40}\d{4}\)", "", text)
    text = re.sub(r"[A-Za-z]+,\s*\d{4}", "", text)
    text = re.sub(r"([A-Za-z]+\s*,){2,}.*", "", text)
    text = re.sub(r"(Universitas|Fakultas|Program Studi|Jurusan|Departemen).*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def remove_duplicate_paragraphs(text: str) -> str:
    """
    Menghapus paragraf atau kalimat yang muncul dua kali (duplikasi PDF).
    Cocok untuk PDF skripsi yang layer text-nya double.
    """
    if not text:
        return text

    paras = [p.strip() for p in text.split("\n") if p.strip()]
    unique = []
    seen = set()

    for p in paras:
        key = p[:120].lower()  # fingerprint pendek
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return "\n".join(unique)
    
def remove_bab_intro_paragraph(text: str) -> str:
    """
    Menghapus paragraf pembuka seperti:
    - 'Bab ini menguraikan...'
    - 'Bab X membahas...'
    - 'Bab ini akan menjelaskan...'
    dan membuang paragraf duplikat otomatis.
    """
    if not text:
        return text

    # buang paragraf pembuka deskriptif
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    clean_paragraphs = []
    
    intro_pattern = re.compile(
        r"^\s*(bab\s*(i|ii|iii|iv|v|\d+)?\s*(ini)?\s*(akan\s+)?"
        r"(membahas|menguraikan|menjelaskan|memaparkan|menjabarkan))",
        flags=re.IGNORECASE
    )

    for p in paragraphs:
        if intro_pattern.search(p):
            continue
        clean_paragraphs.append(p)

    # hilangkan duplikasi paragraf yang sama
    final_unique = []
    seen = set()
    for p in clean_paragraphs:
        key = p[:80].lower()
        if key not in seen:
            seen.add(key)
            final_unique.append(p)

    return "\n".join(final_unique)
    
# SPLIT BAB
def split_by_bab(text: str):
    # Buang elemen non-bab
    text = re.sub(r"DAFTAR\s+ISI.*?(?=BAB\s+I\b|BAB\s+1\b)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"DAFTAR\s+(GAMBAR|TABEL).*?(?=BAB\s+I\b|BAB\s+1\b)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"DAFTAR PUSTAKA.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"LAMPIRAN.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^.*\.{5,}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*[ivxlcdm]+\s*$", "", text, flags=re.IGNORECASE)

    # Mulai dari BAB I (jika ada)
    m = re.search(r"(BAB\s+(?:I|1)\b.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1)

    # Pecah berdasarkan BAB (angka Romawi atau Arab)
    parts = re.split(r"(?=BAB\s+[IVXLCDM]+\b)", text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        parts = re.split(r"(?=BAB\s+\d+\b)", text, flags=re.IGNORECASE)

    candidates = []
    for idx, p in enumerate(parts):
        p = p.strip()
        if not p or not re.match(r"^BAB\s+(?:[IVXLCDM]+|\d+)\b", p, flags=re.IGNORECASE):
            continue
        lines = p.split("\n", 1)
        if len(lines) < 2:
            continue
        judul = lines[0].strip()
        isi = lines[1].strip()
        candidates.append({"judul": judul, "isi": isi, "pos": idx})

    if not candidates:
        return []

    # Kata kunci teknis yang menaikkan skor (bahasa Indonesia + simbol)
    KEYWORDS = [
        "metode","sintesis","represipitasi","psa","karakteris","karakteriza",
        "imobilis","imobiliza","µpad","μpad","nanokristal","bhb","triptamin",
        "imagej","uv","fluores","emisi","analisis","hasil","pembahas","validasi",
        "dispersi","konsentrasi","kecap","sampling","pengujian","selektivitas"
    ]

    def score_text(t: str) -> int:
        s = 0
        low_t = t.lower()
        # dasar: panjang
        s += min(len(low_t), 20000)
        # kata kunci
        key_count = sum(1 for k in KEYWORDS if k in low_t)
        s += key_count * 800
        # jumlah kalimat berguna
        sent_count = len(re.findall(r'[\.!?]', low_t))
        s += min(sent_count, 50) * 50
        # angka/ukur (adanya angka biasanya tanda data atau parameter)
        if re.search(r'\d', low_t):
            s += 500
        # jika ada banyak istilah ilmiah (huruf panjang kata)
        long_word_count = sum(1 for w in re.findall(r'\w+', low_t) if len(w) > 6)
        s += min(long_word_count, 200) * 5
        # penalti jika hanya frasa meta seperti "Bab ini membahas" tanpa kata kunci
        if re.search(r'\bbab\s+\w+\s+membahas', low_t) and key_count == 0 and len(low_t) < 1000:
            s -= 10000
        return s

    # Kelompokkan kandidat berdasarkan judul (BAB I, BAB II, ...)
    groups = {}
    for c in candidates:
        key = re.sub(r'\s+', ' ', c["judul"].upper().strip())
        groups.setdefault(key, []).append(c)

    # Pilih kandidat terbaik per grup (skor tertinggi), simpan pos aslinya
    chosen = []
    for key, items in groups.items():
        best = max(items, key=lambda it: score_text(it["isi"]))
        best["score"] = score_text(best["isi"])
        chosen.append(best)

    # Urutkan berdasarkan posisi terawal kemunculan di dokumen
    chosen.sort(key=lambda x: x["pos"])

    # Final cleaning: buang yang sangat pendek dan tidak informatif
    final = []
    for ch in chosen:
        isi_bersih = re.sub(r'\s+', ' ', ch["isi"]).strip()
        # jika sangat pendek dan tidak mengandung kata kunci penting, skip
        if len(isi_bersih) < 400 and all(k not in isi_bersih.lower() for k in KEYWORDS):
            print(f"{Fore.YELLOW}[FILTER]{Style.RESET_ALL} Menghapus {ch['judul']} (terlalu pendek/tidak teknis).")
            continue
        # Hapus paragraf intro “Bab ini membahas …”
        isi_final = remove_bab_intro_paragraph(ch["isi"])
        final.append({"judul": ch["judul"], "isi": isi_final})

    return final

# UTIL: Tokenisasi & Ringkasan Ekstraktif Lokal
STOPWORDS = set("yang dan di ke dari untuk pada adalah dengan dalam ini itu serta juga tidak dapat atau oleh bagi agar sudah akan para sebagai tersebut karena maka sehingga terhadap serta olehnya".split())
PUNCT = str.maketrans("", "", string.punctuation)

def tokenize(text: str):
    return [w for w in text.lower().translate(PUNCT).split() if w not in STOPWORDS and len(w) > 2]

def split_sentences(text: str):
    sents = re.split(r"(?<=[\.\?\!])\s+(?=[A-Za-z0-9])", text.strip())
    return [s.strip() for s in sents if s.strip()]

def summarize_text_extractive(text: str, max_sent: int = 8) -> str:
    sents = split_sentences(text)
    if not sents: return ""
    sent_tokens = [tokenize(s) for s in sents]
    df = Counter()
    for t in sent_tokens: df.update(set(t))
    N = len(sents)
    scores = []
    for i, toks in enumerate(sent_tokens):
        score = sum((cnt / (1 + len(toks))) * (math.log((N + 1) / (1 + df[w])) + 1)
                    for w, cnt in Counter(toks).items())
        if i < max(3, int(N * 0.1)): score *= 1.15
        scores.append(score)
    top_idx = sorted(range(N), key=lambda i: scores[i], reverse=True)[:max_sent]
    return " ".join([sents[i] for i in sorted(top_idx)])

# PROMPT TEMPLATE
SUM_PROMPT_TEMPLATE = (
    "Tugas kamu adalah merangkum sebuah BAB dari skripsi secara akademik, ringkas, dan tidak repetitif.\n\n"
    "⚠️ ATURAN PENTING:\n"
    "- Jangan mengulang teks dari input.\n"
    "- Jangan membuat dua paragraf yang maknanya sama.\n"
    "- Jangan menyebut 'Bab ini membahas...' atau kalimat pembuka deskriptif.\n"
    "- Hilangkan referensi, kutipan tahun, nomor tabel/gambar, nama lembaga.\n"
    "- Ambil hanya inti ilmiah.\n\n"
    "FORMAT WAJIB:\n"
    "1–2 paragraf ringkasan ilmiah sesuai fungsi BAB:\n"
    "- BAB I → latar belakang + masalah + tujuan\n"
    "- BAB II → teori penting + konsep utama + kerangka teori\n"
    "- BAB III → metode + data + analisis\n"
    "- BAB IV → hasil + analisis pembahasan\n"
    "- BAB V → kesimpulan + saran\n\n"
    "Gunakan bahasa ilmiah yang mengalir dan padat.\n\n"
    "TEKS SUMBER:\n\"\"\"{content}\"\"\"\n\n"
    "RINGKASAN:"
)

# OLLAMA CLIENT DENGAN LOG WARNA
def _ollama_generate(prompt: str) -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception as e:
        print(f"{Fore.RED}[OLLAMA ERROR]{Style.RESET_ALL} {e}")
        return ""

async def ollama_summarize_async(content: str, semaphore: asyncio.Semaphore) -> str:
    prompt = SUM_PROMPT_TEMPLATE.format(content=content)
    attempt = 0
    while True:
        attempt += 1
        try:
            async with semaphore:
                start = time.perf_counter()
                result = await asyncio.to_thread(_ollama_generate, prompt)
                elapsed = time.perf_counter() - start
            if result:
                print(f"{Fore.GREEN}[OLLAMA OK]{Style.RESET_ALL} Ringkasan selesai dalam {elapsed:.1f}s (percobaan ke-{attempt})")
                return result
            raise RuntimeError("Empty response from Ollama.")
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), 8.0)
                print(f"{Fore.YELLOW}[RETRY]{Style.RESET_ALL} Ollama gagal (percobaan ke-{attempt}): {e}. Menunggu {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                print(f"{Fore.RED}[FALLBACK]{Style.RESET_ALL} Semua percobaan gagal, pakai ringkasan lokal (TF-IDF).")
                return summarize_text_extractive(content, max_sent=7)

# RINGKAS PDF PER BAB
def compress_for_prompt(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    base_k = 10 + min(4, len(text) // 20000)
    extract = summarize_text_extractive(text, max_sent=base_k)
    return extract[:max_chars]

async def summarize_sections_parallel(sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _process(sec):
        teks = (sec.get("isi") or "").strip()
        if len(teks) < 80:
            return {"judul": sec["judul"], "ringkasan_bab": ""}

        # bersihkan repetisi, buang intro BAB
        paragraphs = [p.strip() for p in teks.split("\n") if len(p.strip()) > 40]
        isi_bersih = remove_bab_intro_paragraph("\n".join(paragraphs))
        isi_bersih = clean_reference_noise(isi_bersih)

        # kompres jika > batas
        isi_kompres = compress_for_prompt(isi_bersih, MAX_INPUT_CHARS)

        # panggil Ollama
        summary = await ollama_summarize_async(isi_kompres, semaphore)
        summary = summary.strip()

        # bersihkan output LLM dari repetisi dua paragraf sama
        out_paras = [p.strip() for p in summary.split("\n") if p.strip()]
        dedup = []
        seen = set()

        for p in out_paras:
            key = re.sub(r"\s+", " ", p.lower())[:90]  # normalisasi fingerprint
            if key not in seen:
                seen.add(key)
                dedup.append(p)
    
        final_summary = "\n\n".join(dedup)
        return {"judul": sec["judul"], "ringkasan_bab": final_summary}

    return await asyncio.gather(*[asyncio.create_task(_process(sec)) for sec in sections])

def detect_non_thesis(text: str) -> bool:
    if not text or len(text) < 1000: return True
    t = text.lower()
    bab_count = len(re.findall(r"\b(bab\s+(i|ii|iii|iv|v|1|2|3|4|5))\b", t))
    if bab_count < 2: return True
    keywords = ["pendahuluan","tinjauan pustaka","metodologi","hasil","kesimpulan","rumusan masalah","tujuan"]
    if sum(1 for kw in keywords if kw in t) < 3: return True
    if any(x in t for x in ["invoice","laporan keuangan","brosur","sertifikat"]): return True
    return False

async def summarize_pdf_per_bab(path: str):
    raw = read_pdf_text(path)
    if not raw.strip():
        return {"file": os.path.basename(path), "sections": [], "note": "File kosong atau tidak dapat dibaca."}

    raw = remove_duplicate_paragraphs(raw)
    raw = clean_reference_noise(raw)

    if detect_non_thesis(raw):
        return {"file": os.path.basename(path), "sections": [], "note": "File ini tampaknya bukan skripsi atau tugas akhir."}
    sections = split_by_bab(raw)
    if not sections: sections = [{"judul": "BAB I", "isi": raw}]
    results = await summarize_sections_parallel(sections)
    return {"file": os.path.basename(path), "sections": results}

# EKSPOR DOCX & PDF
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def export_all(data, out_docx, out_pdf):
    doc = Document()
    doc.add_heading("Ringkasan Per Bab (Ollama)", 0)
    doc.add_paragraph(f"File: {data['file']}")
    for sec in data["sections"]:
        doc.add_heading(sec["judul"], level=1)
        doc.add_paragraph(sec["ringkasan_bab"] or "")
    doc.save(out_docx)

    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(out_pdf, pagesize=A4)
    elements = [Paragraph("Ringkasan Per Bab (Ollama)", styles['Title']),
                Paragraph(f"File: {data['file']}", styles['Normal']),
                Spacer(1, 12)]
    for sec in data["sections"]:
        elements.append(Paragraph(sec["judul"], styles['Heading2']))
        elements.append(Paragraph(sec["ringkasan_bab"] or "", styles['Normal']))
        elements.append(Spacer(1, 12))
    pdf.build(elements)

# FASTAPI APP
app = FastAPI(title="DocuSum AI (Ollama)", version="9.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
BASE_URL = "https://docusum.onrender.com"

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF diperbolehkan.")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    try:
        hasil = await summarize_pdf_per_bab(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal meringkas: {e}")
    if hasil.get("note"):
        return {"success": False, "message": hasil["note"], "file": hasil["file"]}
    docx_path = str(file_path.with_suffix(".docx"))
    pdf_path = str(file_path.with_suffix(".summary.pdf"))
    export_all(hasil, docx_path, pdf_path)
    hasil["download_docx"] = f"{BASE_URL}/api/download/{Path(docx_path).name}"
    hasil["download_pdf"] = f"{BASE_URL}/api/download/{Path(pdf_path).name}"
    return {"success": True, "data": hasil}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    return FileResponse(file_path, filename=filename)

# KOMENTAR GLOBAL
COMMENTS_FILE = Path("comments.json")

def load_comments() -> list:
    if COMMENTS_FILE.exists():
        try:
            with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_comments(comments: list):
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)

@app.get("/api/comments")
async def get_comments():
    return load_comments()

@app.post("/api/comments")
async def post_comment(comment: Dict[str, str]):
    name = (comment.get("name") or "").strip()
    text = (comment.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Komentar tidak boleh kosong.")
    new_comment = {"name": name or "Anonim", "text": text, "time": datetime.utcnow().isoformat() + "Z"}
    comments = load_comments()
    comments.append(new_comment)
    save_comments(comments)
    return {"success": True, "comment": new_comment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



