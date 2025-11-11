// script.js
// Toast Manager
class ToastManager {
    static showToast(message, type = 'info') {
        const existingToast = document.querySelector('.toast');
        if (existingToast) existingToast.remove();

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${this.getIcon(type)} ${message}</span>`;
        document.body.appendChild(toast);

        setTimeout(() => toast.remove(), 3000);
    }

    static getIcon(type) {
        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
        return icons[type] || '📄';
    }
}

class FileUploader {
    constructor() {
        this.uploadedFiles = [];
        this.processedResults = [];
        this.BASE_API = location.hostname.includes("localhost")
            ? "http://localhost:8000"
            : "https://docusum.onrender.com";

        this.init();
    }

    init() {
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');
        this.browseButton = document.getElementById('browseButton');
        this.uploadList = document.getElementById('uploadList');
        this.processBtn = document.getElementById('processBtn');
        this.clearBtn = document.getElementById('clearBtn');

        this.setupEventListeners();
        this.updateEmptyState();
        this.updateProcessButton();
    }

    setupEventListeners() {
        this.browseButton.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', e => {
            this.handleFiles(e.target.files);
            this.fileInput.value = '';
        });

        this.dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            this.dropZone.classList.add('dragover');
        });

        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('dragover');
        });

        this.dropZone.addEventListener('drop', e => {
            e.preventDefault();
            this.dropZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        this.processBtn.addEventListener('click', () => this.processFiles());
        this.clearBtn.addEventListener('click', () => this.clearAllFiles());
    }

    handleFiles(files) {
        let added = 0;
        for (let file of files) {
            if (this.validateFile(file)) {
                this.addFileToQueue(file);
                added++;
            }
        }
        if (added > 0) ToastManager.showToast(`${added} file ditambahkan`, 'success');
    }

    validateFile(file) {
        if (!file.type.includes('pdf')) {
            ToastManager.showToast('Hanya file PDF yang diperbolehkan!', 'error');
            return false;
        }
        if (this.uploadedFiles.some(f => f.file.name === file.name)) {
            ToastManager.showToast(`File "${file.name}" sudah ada!`, 'warning');
            return false;
        }
        return true;
    }

    addFileToQueue(file) {
        const id = Date.now().toString();
        const fileItem = { id, file, status: 'uploading' };
        this.uploadedFiles.push(fileItem);
        this.renderFileItem(fileItem);
        setTimeout(() => this.updateFileStatus(id, 'completed'), 800);
    }

    renderFileItem(fileItem) {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.id = `file-${fileItem.id}`;
        div.innerHTML = `
            <div class="file-info">
                <div class="file-icon">📄</div>
                <div class="file-details">
                    <h4>${fileItem.file.name}</h4>
                    <span>${(fileItem.file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
            </div>
            <div class="file-status">
                <span class="status-text">Mengupload...</span>
            </div>`;
        this.uploadList.appendChild(div);
        this.updateProcessButton();
    }

    updateFileStatus(id, status) {
        const file = this.uploadedFiles.find(f => f.id === id);
        if (file) {
            file.status = status;
            document.querySelector(`#file-${id} .status-text`).textContent =
                status === 'completed' ? '✅ Selesai' : '❌ Error';
        }
        this.updateProcessButton();
    }

    updateProcessButton() {
        this.processBtn.disabled = !this.uploadedFiles.every(f => f.status === 'completed');
    }

    updateEmptyState() {
        this.uploadList.classList.toggle('empty', this.uploadedFiles.length === 0);
    }

    clearAllFiles() {
        this.uploadedFiles = [];
        this.uploadList.innerHTML = '';
        this.updateProcessButton();
        ToastManager.showToast('Daftar file dibersihkan', 'info');
    }

    async processFiles() {
        this.processBtn.disabled = true;
        this.processBtn.textContent = '⏳ Memproses...';
        this.processedResults = [];

        for (const fileItem of this.uploadedFiles) {
            const result = await this.sendToBackend(fileItem.file);
            if (result) this.processedResults.push(result);
        }

        sessionStorage.setItem('docuSumResults', JSON.stringify(this.processedResults));
        window.location.href = "result.html";
    }

    async sendToBackend(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            ToastManager.showToast(`Mengirim "${file.name}" ke server...`, "info");
            const response = await fetch(`${this.BASE_API}/api/upload`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            // ⚠️ Jika bukan skripsi → tampilkan pesan tapi tetap simpan hasil
            if (!data.success) {
                ToastManager.showToast(data.message || "Bukan dokumen skripsi", "warning");
                return {
                    file: data.file,
                    note: data.message || "Bukan dokumen skripsi",
                    sections: []
                };
            }

            ToastManager.showToast(`"${file.name}" selesai ✅`, "success");
            return data.data; // hanya hasil ringkasan

        } catch (error) {
            console.error("Error:", error);
            ToastManager.showToast(`Gagal memproses "${file.name}" ❌`, "error");
            return null;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => new FileUploader());
