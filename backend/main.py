# main.py
import re, os, string, math, asyncio, time, json, requests
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- CONFIG
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:70b")

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "6"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.6"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "40000"))

# --- UTIL: basic cleaning helpers
BLACKBOX = ["■", "□", "▯", "█", "�"]
PUNCT = str.maketrans("", "", string.punctuation)
STOPWORDS = set("yang dan di ke dari untuk pada adalah dengan dalam ini itu serta juga tidak dapat atau oleh bagi agar sudah akan para sebagai tersebut karena maka sehingga terhadap serta olehnya".split())

def _enough_text(text: str, min_chars: int = 200) -> bool:
    return bool(text and len(text.strip()) >= min_chars)

def clean_text(text: str) -> str:
    if not text:
        return ""
    # normalize newlines & remove weird boxes
    for b in BLACKBOX:
        text = text.replace(b, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # join hyphenated linebreaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # multiple newlines -> two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # remove typical captions and table markers
    text = re.sub(r"(Gambar|Gbr|Tabel)\s*\d+(\.\d+)*", "", text, flags=re.IGNORECASE)
    # remove page headers/footers (simple heuristics: lines with less than 6 words and lots of uppercase)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if len(s.split()) <= 6 and sum(1 for c in s if c.isupper()) > len(s) * 0.6:
            continue
        lines.append(s)
    text = "\n".join(lines)
    # collapse spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def clean_reference_noise(text: str) -> str:
    if not text:
        return ""
    # remove urls, citation footnotes, emails
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    # remove short parenthetical citations like (Smith et al., 2020) or (2020)
    text = re.sub(r"\([A-Za-z][^()]{0,40}\d{4}\)", "", text)
    text = re.sub(r"\(\d{4}\)", "", text)
    # university / department lines
    text = re.sub(r"(Universitas|Fakultas|Program Studi|Jurusan|Departemen).*", "", text, flags=re.IGNORECASE)
    # remove long numeric sequences that are likely page/ids
    text = re.sub(r"\b\d{6,}\b", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def remove_duplicate_paragraphs(text: str) -> str:
    if not text:
        return text
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    seen = set()
    out = []
    for p in paras:
        key = p[:120].lower()
        if key in seen: 
            continue
        seen.add(key)
        out.append(p)
    return "\n".join(out)

def remove_subbab(text: str) -> str:
    """
    Remove sub-section headings like '3.1', '3.1.2', or '3.1. Title' to reduce noise.
    Also collapse enumerated lists that break sentences.
    """
    # remove lines that start with numbering like '3.1' or 'a.' with short titles
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if re.match(r"^\d+(\.\d+)+\s*-?", s):  # 3.1, 2.1.3
            # drop short headings
            if len(s.split()) <= 6:
                continue
            else:
                # remove leading numbering
                s = re.sub(r"^\d+(\.\d+)+\s*[-.:]?\s*", "", s)
        # remove small enumerations "a) text"
        s = re.sub(r"^[a-z]\)\s+", "", s)
        lines.append(s)
    return "\n".join(lines)

# --- SPLIT BY BAB (robust)
def split_by_bab(text: str) -> List[Dict[str, str]]:
    """
    Returns list of {'judul': 'BAB I', 'isi': '...'}.
    Tries ROMAN & arabic numbering and falls back to heuristic splits (based on common 'BAB').
    """
    if not text:
        return []
    t = text
    # remove daftar isi block if present
    t = re.sub(r"(?is)DAFTAR\s+ISI.*?(?=(BAB\s+I\b|BAB\s+1\b))", "", t)
    t = re.sub(r"(?is)DAFTAR\s+(GAMBAR|TABEL).*?(?=(BAB\s+I\b|BAB\s+1\b))", "", t)
    t = re.sub(r"(?is)DAFTAR PUSTAKA.*", "", t)
    t = re.sub(r"(?is)LAMPIRAN.*", "", t)
    # find first BAB I
    m = re.search(r"(BAB\s+(?:I|1)\b.*)", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1)
    # split by BAB markers (roman or arabic)
    parts = re.split(r"(?=^BAB\s+(?:[IVXLCDM]+|\d+)\b)", t, flags=re.IGNORECASE | re.MULTILINE)
    if len(parts) <= 1:
        # fallback: split by top-level headings "BAB" anywhere
        parts = re.split(r"(?i)(?=^BAB\s+)", t, flags=re.MULTILINE)
    candidates = []
    for idx, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        # get heading (first line)
        lines = p.split("\n", 1)
        if len(lines) == 1:
            judul = lines[0].strip()
            isi = ""
        else:
            judul = lines[0].strip()
            isi = lines[1].strip()
        # normalize title to "BAB X"
        title_match = re.match(r"(BAB\s+(?:[IVXLCDM]+|\d+)\b.*)", judul, flags=re.IGNORECASE)
        if not title_match:
            continue
        candidates.append({"judul": judul, "isi": isi, "pos": idx})
    # score and pick best per group with same roman/arabic number (some PDFs duplicate)
    def score_text(tstr: str) -> int:
        score = 0
        low = (tstr or "").lower()
        score += min(len(low), 20000)
        # important keywords that likely indicate technical content
        kws = ["metode", "metodologi", "hasil", "analisis", "eksperimen", "data", "pustaka", "kesimpulan", "penelitian"]
        score += sum(800 for k in kws if k in low)
        sent_count = len(re.findall(r'[\.!?]', low))
        score += min(sent_count, 80) * 40
        if re.search(r'\d', low):
            score += 300
        long_words = sum(1 for w in re.findall(r'\w+', low) if len(w) > 7)
        score += min(long_words, 500) * 3
        return score
    # group by normalized BAB id (e.g., "BAB I")
    groups = {}
    for c in candidates:
        norm = re.sub(r"\s+", " ", re.match(r"(BAB\s+(?:[IVXLCDM]+|\d+))", c["judul"], flags=re.IGNORECASE).group(1).upper())
        groups.setdefault(norm, []).append(c)
    chosen = []
    for k, items in groups.items():
        best = max(items, key=lambda x: score_text(x["isi"]))
        best["score"] = score_text(best["isi"])
        chosen.append(best)
    chosen.sort(key=lambda x: x["pos"])
    # final filter: if a section is too short, keep but mark
    final = []
    for ch in chosen:
        isi = ch["isi"]
        isi = remove_subbab(isi)
        isi = remove_duplicate_paragraphs(isi)
        isi = clean_reference_noise(isi)
        # drop if less than 120 chars and not likely important
        if len(re.sub(r'\s+', ' ', isi)) < 120:
            # still include but with original minimal content
            final.append({"judul": ch["judul"], "isi": isi})
            continue
        final.append({"judul": ch["judul"], "isi": isi})
    return final

# --- simple extractive summarizer (fallback)
def tokenize(text: str):
    return [w for w in text.lower().translate(PUNCT).split() if w not in STOPWORDS and len(w) > 2]

def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])", text.strip())
    return [s.strip() for s in sents if s.strip()]

def summarize_text_extractive(text: str, max_sent: int = 6) -> str:
    sents = split_sentences(text)
    if not sents:
        # fallback naive: return first 3 lines
        return "\n".join(text.split("\n")[:3])
    tokens = [tokenize(s) for s in sents]
    df = Counter()
    for t in tokens:
        df.update(set(t))
    N = len(sents)
    scores = []
    for i, toks in enumerate(tokens):
        if not toks:
            scores.append(0.0); continue
        sc = sum((1.0 / (1 + len(toks))) * (math.log((N + 1) / (1 + df[w])) + 1) for w in Counter(toks))
        # position boost
        if i < max(3, int(N * 0.08)):
            sc *= 1.15
        scores.append(sc)
    top = sorted(range(N), key=lambda i: scores[i], reverse=True)[:max_sent]
    top_sorted = sorted(top)
    return " ".join([sents[i] for i in top_sorted])

# --- PROMPT (much stricter)
SUM_PROMPT_TEMPLATE = (
    "Kamu adalah asisten ringkasan akademik. "
    "Input: TEKS SUMBER (isi satu BAB skripsi). "
    "Tugas: buat 2 bagian terpisah: TLDR (satu kalimat sangat ringkas) dan RINGKASAN (1 paragraf, maksimal 4 kalimat).\n\n"
    "KETENTUAN (WAJIB):\n"
    "- TLDR: tepat 1 kalimat, langsung ke inti (latar/tujuan/metode/hasil/kesimpulan sesuai fungsi BAB). "
    "- RINGKASAN: 1 paragraf, padat, gunakan bahasa ilmiah sederhana, jangan ulang kata persis dari Teks Sumber.\n"
    "- Jangan masukkan kutipan, referensi, atau daftar pustaka.\n"
    "- Jangan mulai TLDR dengan 'Bab ini'. Jangan pakai frasa 'bab ini'.\n"
    "- Jika teks sumber sangat metodologis (BAB III) fokuskan ringkasan pada metode, desain, data & alat; jika BAB IV fokus pada hasil dan interpretasi; jika BAB I fokus pada masalah & tujuan.\n"
    "- Pastikan TLDR dan RINGKASAN berbeda sepenuhnya.\n\n"
    "OUTPUT WAJIB (format EXACT):\n"
    "TLDR:\n"
    "<satu kalimat>\n\n"
    "RINGKASAN:\n"
    "<satu paragraf (1-4 kalimat)>\n\n"
    "TEKS SUMBER:\n\"\"\"\n{content}\n\"\"\"\n"
)

def _ollama_generate(payload_json: dict) -> str:
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload_json, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # expect {"response": "..."} for many local deployments
        if isinstance(data, dict):
            return (data.get("response") or data.get("text") or "") .strip()
        return str(data).strip()
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ""

async def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.1
        }
    }
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, _ollama_generate, payload)
            if out and out.strip():
                return out
            await asyncio.sleep(RETRY_BASE_DELAY * attempt)
        except Exception:
            await asyncio.sleep(RETRY_BASE_DELAY * attempt)
    return ""

# --- process sections in parallel with semaphore
async def summarize_sections_parallel(sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process(sec: Dict[str, str]) -> Dict[str, Any]:
        judul = sec.get("judul", "BAB")
        isi_raw = (sec.get("isi") or "").strip()
        if not isi_raw:
            return {"judul": judul, "ringkasan_bab": "", "tldr": ""}

        # clean & compress input to reasonable size for prompt
        isi = clean_text(isi_raw)
        isi = clean_reference_noise(isi)
        isi = remove_subbab(isi)
        isi = remove_duplicate_paragraphs(isi)
        if len(isi) > MAX_INPUT_CHARS:
            # use extractive to compress
            isi = summarize_text_extractive(isi, max_sent=12)

        prompt = SUM_PROMPT_TEMPLATE.format(content=isi)

        async with sem:
            llm_out = await call_ollama(prompt)

        llm_out = (llm_out or "").strip()
        # fallback: if LLM returns empty or not in format, use extractive summarizer
        tldr_text = ""
        ringkasan_text = ""

        def parse_llm(text: str):
            tldr = ""
            ring = ""
            # try to extract fields
            m_tldr = re.search(r"TLDR:\s*(.+?)(?:\n\s*\n|$)", text, flags=re.S)
            if m_tldr:
                tldr = m_tldr.group(1).strip().replace("\n", " ")
            m_ring = re.search(r"RINGKASAN:\s*(.+?)(?:\n\s*\n|$)", text, flags=re.S)
            if m_ring:
                ring = m_ring.group(1).strip()
            return tldr, ring

        if llm_out:
            tldr_text, ringkasan_text = parse_llm(llm_out)

        # if missing or too long/too short, fallback to extractive
        if not ringkasan_text or len(ringkasan_text.split()) < 8:
            # build a 1-paragraph extractive summary
            ringkasan_text = summarize_text_extractive(isi, max_sent=4)
        if not tldr_text or len(tldr_text.split()) < 3:
            # make a short TLDR from the extractive summary: take first sentence
            sents = split_sentences(ringkasan_text)
            tldr_text = sents[0] if sents else (ringkasan_text.split(".")[0] + ".")

        # ensure TLDR is one sentence
        tldr_text = re.sub(r"\s+", " ", tldr_text).strip()
        if not tldr_text.endswith("."):
            tldr_text = tldr_text.split(".")[0].strip() + "."

        # post-clean: avoid repeating "Bab ini..."
        tldr_text = re.sub(r"^[Bb]ab\s+ini[:,]?\s*", "", tldr_text).strip()
        ringkasan_text = re.sub(r"^[Bb]ab\s+ini[:,]?\s*", "", ringkasan_text).strip()

        # final bounding: ringkasan max 3 sentences
        ring_sents = split_sentences(ringkasan_text)
        ringkasan_text = " ".join(ring_sents[:3]) if ring_sents else ringkasan_text

        return {"judul": judul, "ringkasan_bab": ringkasan_text, "tldr": tldr_text}

    tasks = [asyncio.create_task(process(sec)) for sec in sections]
    results = await asyncio.gather(*tasks)
    return results

# --- high-level pipeline
async def summarize_pdf_per_bab(path: str) -> Dict[str, Any]:
    raw = read_pdf_text(path)
    if not raw or not raw.strip():
        return {"file": os.path.basename(path), "sections": [], "note": "File kosong atau tidak dapat dibaca."}
    raw = clean_text(raw)
    raw = remove_duplicate_paragraphs(raw)
    raw = clean_reference_noise(raw)

    # quick check for thesis-like content
    txt_lower = raw.lower()
    if len(raw) < 800 or sum(1 for k in ["pendahuluan", "tinjauan pustaka", "metodologi", "hasil", "kesimpulan"] if k in txt_lower) < 2:
        # still attempt but mark note
        note = "Dokumen tampak singkat atau tidak standar (bukan skripsi lengkap)."
    else:
        note = ""

    sections = split_by_bab(raw)
    if not sections:
        # fallback: treat whole doc as one section
        sections = [{"judul": "BAB I", "isi": raw}]

    results = await summarize_sections_parallel(sections)
    out = {"file": os.path.basename(path), "sections": results}
    if note:
        out["note"] = note
    return out

# --- PDF/Docx export
def export_all(data: dict, out_docx: str, out_pdf: str):
    doc = Document()
    doc.add_heading("Ringkasan Per Bab", 0)
    doc.add_paragraph(f"File: {data.get('file','')}")
    for sec in data.get("sections", []):
        doc.add_heading(sec.get("judul", ""), level=1)
        doc.add_paragraph("TLDR: " + (sec.get("tldr") or ""))
        doc.add_paragraph(sec.get("ringkasan_bab") or "")
    doc.save(out_docx)

    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(out_pdf, pagesize=A4)
    elements = [Paragraph("Ringkasan Per Bab", styles["Title"]), Paragraph(f"File: {data.get('file','')}", styles["Normal"]), Spacer(1, 12)]
    for sec in data.get("sections", []):
        elements.append(Paragraph(sec.get("judul", ""), styles["Heading2"]))
        elements.append(Paragraph("TLDR: " + (sec.get("tldr") or ""), styles["Normal"]))
        elements.append(Paragraph(sec.get("ringkasan_bab") or "", styles["Normal"]))
        elements.append(Spacer(1, 12))
    pdf.build(elements)

# --- PDF text extraction using several backends (attempt order)
def read_pdf_text(path: str) -> str:
    text = ""
    # 1) PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(path)
        pages = []
        for p in doc:
            pages.append(p.get_text())
        text = "\n".join(pages)
        if _enough_text(text):
            return clean_text(text)
    except Exception:
        pass
    # 2) pdfminer
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path) or ""
        if _enough_text(text):
            return clean_text(text)
    except Exception:
        pass
    # 3) PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        text = "\n".join(pages)
        if _enough_text(text):
            return clean_text(text)
    except Exception:
        pass
    # 4) OCR fallback (pdf2image + pytesseract)
    try:
        from pdf2image import convert_from_path
        import pytesseract
        pages = convert_from_path(path, dpi=300)
        ocr_text = ""
        for pg in pages:
            ocr_text += pytesseract.image_to_string(pg, lang="eng+ind") + "\n"
        return clean_text(ocr_text)
    except Exception:
        pass
    return clean_text(text)

# --- FastAPI app
app = FastAPI(title="DocuSum AI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF diperbolehkan.")
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        hasil = await summarize_pdf_per_bab(str(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal meringkas: {e}")
    if hasil.get("note") and not hasil.get("sections"):
        return {"success": False, "message": hasil["note"], "file": hasil["file"]}
    docx_path = str(path.with_suffix(".docx"))
    pdf_path = str(path.with_suffix(".summary.pdf"))
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

# comments endpoints (unchanged)
COMMENTS_FILE = Path("comments.json")
def load_comments() -> list:
    if COMMENTS_FILE.exists():
        try:
            return json.loads(COMMENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_comments(comments: list):
    COMMENTS_FILE.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")

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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
