const API_URL = "http://localhost:8000";

let currentFile = null;

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

        // Log QA report for debugging
        if (data.qa) {
            console.log('[QA Report]', data.qa);
        }

        // Check QA (use 'ok' field)
        const qaPass = data.ok !== undefined ? data.ok : data.qa_pass; // Backward compatible

        if (!qaPass) {
            // CLEAR ALL PREVIOUS RESULTS
            clearResults();

            // Show simple rejection message
            showAlert("❌ QA Rejected", "error");
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

    // Distribution
    const dist = data.distribution;
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

    downloadBtn.href = img.src;
    downloadBtn.classList.remove('hidden');

    // Scroll to results
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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

        if (data.training_curves) {
            document.getElementById('chart-curves').src = `${API_URL}${data.training_curves}?t=${ts}`;
        }

        if (data.confusion_matrix) {
            document.getElementById('chart-cm').src = `${API_URL}${data.confusion_matrix}?t=${ts}`;
        }
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

        // Total predictions
        document.getElementById('dash-total').textContent = history.length;

        // Calculate averages by mode
        const yoloItems = history.filter(h => h.mode === 'YOLO');
        const effnetItems = history.filter(h => h.mode === 'EfficientNet');
        const hybridItems = history.filter(h => h.mode === 'Hybrid');

        const avgYolo = yoloItems.length ?
            (yoloItems.reduce((sum, h) => sum + h.confidence, 0) / yoloItems.length * 100).toFixed(0) + '%' : '--';
        const avgEffnet = effnetItems.length ?
            (effnetItems.reduce((sum, h) => sum + h.confidence, 0) / effnetItems.length * 100).toFixed(0) + '%' : '--';
        const avgHybrid = hybridItems.length ?
            (hybridItems.reduce((sum, h) => sum + h.confidence, 0) / hybridItems.length * 100).toFixed(0) + '%' : '--';

        document.getElementById('dash-yolo').textContent = avgYolo;
        document.getElementById('dash-effnet').textContent = avgEffnet;
        document.getElementById('dash-hybrid').textContent = avgHybrid;

    } catch (e) {
        console.warn("Dashboard load failed:", e);
    }
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
