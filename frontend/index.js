const API_URL = "http://localhost:8000";

let currentFile = null;
let batchFiles = [];
let lastPredictionData = null;
let originalImageBase64 = null;

// ==================== NAVIGATION ====================
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    // Show selected page
    document.getElementById(pageId + '-page').classList.add('active');

    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.nav-item').classList.add('active');

    // Load data for specific pages
    if (pageId === 'analytics') {
        loadAnalytics();
    } else if (pageId === 'dashboard') {
        loadDashboard();
    }
}

// ==================== DRAG & DROP ====================
const dropArea = document.getElementById('drop-area');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.style.borderColor = 'var(--primary)';
    });
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.style.borderColor = 'var(--border)';
    });
});

dropArea.addEventListener('drop', handleDrop);
function handleDrop(e) {
    handleFiles(e.dataTransfer.files);
}

// Click to upload
dropArea.addEventListener('click', function (e) {
    if (!e.target.closest('.clear-image-btn') && !e.target.closest('#preview-container')) {
        document.getElementById('fileElem').click();
    }
});

// ==================== FILE HANDLING ====================
function handleFiles(files) {
    const file = files[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showAlert("Invalid file type. Please upload JPG, PNG, or WEBP.", "error");
        return;
    }

    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
        showAlert("Image too large. Maximum size is 10MB.", "error");
        return;
    }

    currentFile = file;
    // For PDF report fallback
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImageBase64 = e.target.result;
    };
    reader.readAsDataURL(file);

    showPreview(file);
    document.getElementById('predict-btn').disabled = false;
    hideAlert();
}

function showPreview(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function (e) {
        document.getElementById('preview-img').src = e.target.result;
        document.getElementById('preview-container').classList.remove('hidden');
    }
}

function clearImage(e) {
    e.stopPropagation();
    currentFile = null;
    document.getElementById('preview-img').src = "";
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('predict-btn').disabled = true;
    document.getElementById('fileElem').value = "";
    hideAlert();
}

// ==================== ALERTS ====================
function showAlert(message, type = "info") {
    const alert = document.getElementById('qa-alert');
    alert.textContent = message;
    alert.className = `alert ${type}`;
    alert.classList.remove('hidden');
}

function hideAlert() {
    const alert = document.getElementById('qa-alert');
    alert.classList.add('hidden');
}

// ==================== PREDICTION ====================
async function runPrediction() {
    if (!currentFile) return;

    const mode = document.querySelector('input[name="mode"]:checked').value;
    const endpoint = `/predict/${mode}`;

    const btn = document.getElementById('predict-btn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');

    // Loading state
    btn.disabled = true;
    btnText.textContent = "Processing...";
    spinner.classList.remove('hidden');
    hideAlert();
    clearResults(); // Clear previous results immediately

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Server error");
        }

        const data = await response.json();
        lastPredictionData = data; // Store for PDF report

        // Log QA report for debugging
        if (data.qa) {
            console.log('[QA Report]', data.qa);
        }

        // Check QA (use 'ok' field)
        const qaPass = data.ok !== undefined ? data.ok : data.qa_pass; // Backward compatible

        if (!qaPass) {
            // CLEAR ALL PREVIOUS RESULTS
            clearResults();

            // Show rejection reason from server
            const reason = data.message || "QA Rejected";
            showAlert(`❌ ${reason}`, "error");
            console.warn('[QA REJECT]', data.qa);
            return;
        }

        // Success - QA Passed
        showAlert(`✓ ${data.message || data.qa_message || "QA Passed"}`, "success");
        displayResults(data);

    } catch (error) {
        // Clear results on any error
        clearResults();
        showAlert(`Prediction failed: ${error.message}`, "error");
        console.error("Error:", error);
    } finally {
        btn.disabled = false;
        btnText.textContent = "Analyze Maturity";
        spinner.classList.add('hidden');
    }
}

function displayResults(data) {
    const panel = document.getElementById('result-panel');
    panel.classList.remove('hidden');

    // Mode badge
    const badge = document.getElementById('mode-badge');
    badge.textContent = data.mode;

    // Check for agreement boosting
    if (data.mode === 'Hybrid' && data.agreement) {
        badge.textContent = "AGREEMENT BOOSTED";
        badge.className = "badge badge-boosted";
    } else {
        badge.className = `badge badge-${data.mode.toLowerCase().replace('efficientnet', 'efficientnet')}`;
    }

    // Final label
    const labelEl = document.getElementById('res-label');
    labelEl.textContent = data.final_label;

    // Add boosted styling to label if applicable
    if (data.agreement) {
        labelEl.className = `metric-value label-${data.final_label.toLowerCase()} boosted`;
    } else {
        labelEl.className = `metric-value label-${data.final_label.toLowerCase()}`;
    }

    // Confidence
    const confEl = document.getElementById('res-conf');
    confEl.innerHTML = `${(data.final_confidence * 100).toFixed(1)}%`;

    // Add confidence note
    if (data.status_msg) {
        const note = document.createElement('span');
        note.className = "confidence-note";
        note.textContent = data.status_msg;
        confEl.appendChild(note);
    }

    if (data.agreement) {
        confEl.classList.add('boosted');
    } else {
        confEl.classList.remove('boosted');
    }

    // Count
    document.getElementById('res-count').textContent = data.fruits_detected;

    // Class Breakdown (YOLO/Hybrid only)
    const breakdownEl = document.getElementById('class-breakdown');
    breakdownEl.innerHTML = ''; // Clear previous
    breakdownEl.classList.add('hidden');

    const dist = data.distribution;

    if (data.mode !== 'EfficientNet' && data.fruits_detected > 0) {
        breakdownEl.classList.remove('hidden');

        ['Unripe', 'Ripe', 'Overripe'].forEach(cls => {
            const count = dist[cls];
            if (count > 0) {
                const badge = document.createElement('div');
                badge.className = `class-stat ${cls.toLowerCase()}`;
                // Add dot
                const dot = document.createElement('i');
                dot.className = `legend-dot ${cls.toLowerCase()}`;

                const text = document.createTextNode(`${cls}: `);
                const val = document.createElement('span');
                val.textContent = count;

                badge.appendChild(dot);
                badge.appendChild(text);
                badge.appendChild(val);
                breakdownEl.appendChild(badge);
            }
        });
    }
    const total = dist.Unripe + dist.Ripe + dist.Overripe;

    // Toggle thick bar for agreement
    const distBar = document.querySelector('.distribution-bar');
    if (data.mode === 'Hybrid' && data.agreement) {
        distBar.classList.add('thick-bar');
    } else {
        distBar.classList.remove('thick-bar');
    }

    if (total > 0) {
        const unripePct = (dist.Unripe / total) * 100;
        const ripePct = (dist.Ripe / total) * 100;
        const overripePct = (dist.Overripe / total) * 100;

        document.getElementById('bar-unripe').style.width = `${unripePct}%`;
        document.getElementById('bar-ripe').style.width = `${ripePct}%`;
        document.getElementById('bar-overripe').style.width = `${overripePct}%`;

        document.getElementById('bar-unripe').querySelector('.dist-label').textContent = dist.Unripe || '';
        document.getElementById('bar-ripe').querySelector('.dist-label').textContent = dist.Ripe || '';
        document.getElementById('bar-overripe').querySelector('.dist-label').textContent = dist.Overripe || '';
    }

    // Image
    const img = document.getElementById('result-img');
    const downloadBtn = document.getElementById('download-btn');

    if (data.annotated_image) {
        // Use annotated image from backend
        img.src = `data:image/jpeg;base64,${data.annotated_image}`;
    } else {
        // Fallback to input image (EfficientNet default)
        // This ensures filter/boxes aren't shown, just the clean input
        const previewImg = document.getElementById('preview-img');
        if (previewImg) {
            img.src = previewImg.src;
        }
    }

    downloadBtn.onclick = (e) => {
        e.preventDefault();
        downloadPDFReport();
    };
    downloadBtn.innerHTML = "📄 Download Research PDF";
    downloadBtn.classList.remove('hidden');

    // Scroll to results
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function downloadPDFReport() {
    if (!lastPredictionData) return;

    // Ensure original image is included for report fallback
    const reportData = {
        ...lastPredictionData,
        original_image: originalImageBase64
    };

    const btn = document.getElementById('download-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = "⌛ Generating PDF...";
    btn.style.pointerEvents = "none";

    try {
        const response = await fetch(`${API_URL}/predict/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reportData)
        });

        if (!response.ok) throw new Error("Failed to generate PDF");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `CoffeeAI_Report_${new Date().getTime()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error("PDF Export Error:", error);
        alert("Failed to generate PDF report.");
    } finally {
        btn.innerHTML = originalText;
        btn.style.pointerEvents = "auto";
    }
}

function clearResults() {
    document.getElementById('result-panel').classList.add('hidden');
    document.getElementById('result-img').src = ""; // Clear to prevent ghosting
}

// ==================== ANALYTICS ====================
async function loadAnalytics() {
    try {
        const response = await fetch(`${API_URL}/analytics`);
        if (!response.ok) return;

        const data = await response.json();
        const ts = Date.now();

        // Helper to load image
        const loadImage = (imgId, containerId, srcPath) => {
            const img = document.getElementById(imgId);
            const container = document.getElementById(containerId);

            if (!img || !container) return;

            if (srcPath) {
                img.onload = () => {
                    container.classList.remove('error');
                };
                img.onerror = () => {
                    container.classList.add('error');
                };
                img.src = `${API_URL}${srcPath}?t=${ts}`;
            } else {
                container.classList.add('error');
            }
        };

        loadImage('chart-curves', 'container-curves', data.training_curves);
        loadImage('chart-cm', 'container-cm', data.confusion_matrix);

    } catch (e) {
        console.warn("Analytics load failed:", e);
    }
}

// ==================== DASHBOARD ====================
async function loadDashboard() {
    try {
        const response = await fetch(`${API_URL}/data/history?limit=100`);
        if (!response.ok) return;

        const data = await response.json();
        const history = data.history || [];
        const total = data.total || history.length;

        // Total predictions
        document.getElementById('dash-total').textContent = total;

        // Calculate averages by mode (handle both uppercase and lowercase from DB/API)
        const yoloItems = history.filter(h => h.mode && h.mode.toLowerCase() === 'yolo');
        const effnetItems = history.filter(h => h.mode && (h.mode.toLowerCase() === 'effnet' || h.mode.toLowerCase() === 'efficientnet'));
        const hybridItems = history.filter(h => h.mode && h.mode.toLowerCase() === 'hybrid');

        const avgYolo = yoloItems.length ?
            (yoloItems.reduce((sum, h) => sum + (h.final_confidence || 0), 0) / yoloItems.length * 100).toFixed(0) + '%' : '--';
        const avgEffnet = effnetItems.length ?
            (effnetItems.reduce((sum, h) => sum + (h.final_confidence || 0), 0) / effnetItems.length * 100).toFixed(0) + '%' : '--';
        const avgHybrid = hybridItems.length ?
            (hybridItems.reduce((sum, h) => sum + (h.final_confidence || 0), 0) / hybridItems.length * 100).toFixed(0) + '%' : '--';

        document.getElementById('dash-yolo').textContent = avgYolo;
        document.getElementById('dash-effnet').textContent = avgEffnet;
        document.getElementById('dash-hybrid').textContent = avgHybrid;

    } catch (e) {
        console.warn("Dashboard load failed:", e);
    }
}

function exportToCSV() {
    window.location.href = `${API_URL}/data/export/csv`;
}

// ==================== BATCH PROCESSING ====================
function handleBatchFiles(files) {
    const newFiles = Array.from(files).slice(0, 10); // Limit to 10
    batchFiles = newFiles;

    const container = document.getElementById('batch-list-container');
    const countLabel = document.getElementById('batch-count');
    const queue = document.getElementById('batch-queue');

    if (batchFiles.length > 0) {
        container.classList.remove('hidden');
        countLabel.textContent = `${batchFiles.length} images selected`;
        renderBatchQueue();
    } else {
        container.classList.add('hidden');
    }
}

function renderBatchQueue() {
    const queue = document.getElementById('batch-queue');
    queue.innerHTML = '';

    batchFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.style = "display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: rgba(255,255,255,0.03); border-radius: 8px; font-size: 13px;";
        item.innerHTML = `
            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px;">${file.name}</span>
            <span style="color: var(--text-muted); font-size: 11px;">${(file.size / 1024).toFixed(0)} KB</span>
        `;
        queue.appendChild(item);
    });
}

function clearBatch() {
    batchFiles = [];
    document.getElementById('batch-list-container').classList.add('hidden');
    document.getElementById('batch-results-grid').classList.add('hidden');
    document.getElementById('batchFileElem').value = '';
}

async function runBatchPrediction() {
    if (batchFiles.length === 0) return;

    const btn = document.getElementById('batch-run-btn');
    const spinner = btn.querySelector('.spinner');
    const btnText = btn.querySelector('.btn-text');

    btn.disabled = true;
    btnText.textContent = "Processing Batch...";
    spinner.classList.remove('hidden');

    const formData = new FormData();
    batchFiles.forEach(file => {
        formData.append('files', file);
    });

    try {
        const response = await fetch(`${API_URL}/predict/batch`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Batch processing failed");

        const data = await response.json();
        displayBatchResults(data.results);

    } catch (error) {
        showAlert(`Batch failed: ${error.message}`, "error");
    } finally {
        btn.disabled = false;
        btnText.textContent = "Process Batch";
        spinner.classList.add('hidden');
    }
}

function displayBatchResults(results) {
    const grid = document.getElementById('batch-results-grid');
    grid.innerHTML = '';
    grid.classList.remove('hidden');

    results.forEach(res => {
        const card = document.createElement('div');
        card.className = "card";
        card.style = "padding: 1.25rem; border-left: 4px solid " + (res.ok ? "var(--success)" : "var(--error)");

        if (res.ok) {
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                    <h4 style="font-size: 14px; margin: 0; overflow: hidden; text-overflow: ellipsis;">${res.filename}</h4>
                    <span class="badge badge-hybrid">Hybrid</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem;">
                    <div>
                        <label style="font-size: 10px; color: var(--text-muted); display: block;">Label</label>
                        <span style="font-weight: 600; font-size: 15px; color: var(--primary);">${res.final_label}</span>
                    </div>
                    <div>
                        <label style="font-size: 10px; color: var(--text-muted); display: block;">Confidence</label>
                        <span style="font-weight: 600; font-size: 15px;">${(res.final_confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        } else {
            card.innerHTML = `
                <h4 style="font-size: 14px; margin-bottom: 0.5rem;">${res.filename}</h4>
                <p style="color: var(--error); font-size: 12px; margin: 0;">❌ ${res.message || "QA Rejected"}</p>
            `;
        }
        grid.appendChild(card);
    });

    grid.scrollIntoView({ behavior: 'smooth' });
}

// ==================== HEALTH CHECK ====================
async function checkHealth() {
    const statusDot = document.getElementById('server-status');
    const statusText = document.getElementById('status-text');

    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            statusDot.classList.add('online');
            statusText.textContent = 'Server Online';
        }
    } catch {
        statusDot.classList.remove('online');
        statusText.textContent = 'Server Offline';
    }
}

// ==================== INITIALIZATION ====================
checkHealth();
setInterval(checkHealth, 30000);

// Initial data load
loadAnalytics();
loadDashboard();
