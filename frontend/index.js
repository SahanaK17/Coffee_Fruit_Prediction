const API_URL = "http://localhost:8000";

let currentFile = null;

let lastPredictionData = null;
let originalImageBase64 = null;
let currentYieldFile = null;
let yieldOriginalImageBase64 = null;
let lastYieldData = null;

// ==================== NAVIGATION ====================
function showPage(pageId) {
    // The compare page is a fixed overlay outside <main>, handle separately
    const comparePage = document.getElementById('compare-page');
    if (pageId === 'compare') {
        // Hide all normal pages
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        comparePage.style.display = 'block';
    } else {
        comparePage.style.display = 'none';
        // Hide all pages then show the selected one
        document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
        document.getElementById(pageId + '-page').classList.add('active');
    }

    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    if (event && event.target) event.target.closest('.nav-item').classList.add('active');

    // Load data for specific pages
    if (pageId === 'dashboard') loadDashboard();
}
// ==================== PREDICT TAB SWITCHING ====================
function switchPredictTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.predict-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`.predict-tab[data-tab="${tabName}"]`).classList.add('active');

    // Update tab content sections
    document.querySelectorAll('.predict-tab-content').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(tabName + '-section').classList.add('active');
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

// ==================== YIELD UPLOAD (click + drag/drop) ====================
const yieldDropArea = document.getElementById('yield-drop-area');
if (yieldDropArea) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        yieldDropArea.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        yieldDropArea.addEventListener(eventName, () => {
            yieldDropArea.style.borderColor = 'var(--primary)';
        });
    });
    ['dragleave', 'drop'].forEach(eventName => {
        yieldDropArea.addEventListener(eventName, () => {
            yieldDropArea.style.borderColor = 'var(--border)';
        });
    });
    yieldDropArea.addEventListener('drop', (e) => {
        handleYieldFiles(e.dataTransfer.files);
    });
    yieldDropArea.addEventListener('click', function (e) {
        if (!e.target.closest('.clear-image-btn') && !e.target.closest('#yield-preview-container')) {
            document.getElementById('yieldFileElem').click();
        }
    });
}

// ==================== YIELD FILE HANDLING ====================
function handleYieldFiles(files) {
    const file = files[0];
    if (!file) return;

    if (!['image/jpeg', 'image/png', 'image/jpg', 'image/webp'].includes(file.type)) {
        showAlert("Invalid file type for yield analysis.", "error");
        return;
    }

    currentYieldFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        yieldOriginalImageBase64 = e.target.result;
        document.getElementById('yield-preview-img').src = e.target.result;
        document.getElementById('yield-preview-container').classList.remove('hidden');
        document.getElementById('yield-upload-placeholder').classList.add('hidden');
        document.getElementById('yield-predict-btn').disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearYieldImage(e) {
    if (e) e.stopPropagation();
    currentYieldFile = null;
    document.getElementById('yield-preview-img').src = "";
    document.getElementById('yield-preview-container').classList.add('hidden');
    document.getElementById('yield-upload-placeholder').classList.remove('hidden');
    document.getElementById('yield-predict-btn').disabled = true;
    document.getElementById('yieldFileElem').value = "";
}

function resetYieldPrediction() {
    clearYieldImage();
    document.getElementById('yield-result').classList.add('hidden');
    document.getElementById('yield-empty-state').style.display = 'flex';
    hideAlert('yield-alert');
    
    // Scroll back to top of the yield section
    const yieldSection = document.getElementById('yield-section');
    if (yieldSection) {
        yieldSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ==================== ALERTS ====================
function showAlert(message, type = "info", targetId = 'qa-alert') {
    const alertEl = document.getElementById(targetId);
    if (!alertEl) return;
    alertEl.textContent = message;
    alertEl.className = `alert ${type}`;
    alertEl.classList.remove('hidden');
}

function hideAlert(targetId = 'qa-alert') {
    const alertEl = document.getElementById(targetId);
    if (alertEl) alertEl.classList.add('hidden');
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

        // Removed yield-predict-action since Yield is now in its own tab
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

    // Soil Data Display
    const soilDisplay = document.getElementById('soil-data-display');
    if (data.mode === 'Hybrid+Soil' && data.soil_data) {
        soilDisplay.classList.remove('hidden');
        document.getElementById('res-soil-ph').textContent = data.soil_data.ph ? data.soil_data.ph.toFixed(1) : 'N/A';
        document.getElementById('res-soil-clay').textContent = data.soil_data.clay ? data.soil_data.clay.toFixed(1) + ' %' : 'N/A';
        document.getElementById('res-soil-sand').textContent = data.soil_data.sand ? data.soil_data.sand.toFixed(1) + ' %' : 'N/A';
        document.getElementById('res-soil-silt').textContent = data.soil_data.silt ? data.soil_data.silt.toFixed(1) + ' %' : 'N/A';
        document.getElementById('res-soil-soc').textContent = data.soil_data.soc ? data.soil_data.soc.toFixed(1) + ' g/kg' : 'N/A';
    } else {
        soilDisplay.classList.add('hidden');
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

let yieldChart = null;

// ==================== YIELD LOCATION & PREDICTION ====================
function getBrowserLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition((position) => {
            document.getElementById('yield-lat').value = position.coords.latitude.toFixed(8);
            document.getElementById('yield-lon').value = position.coords.longitude.toFixed(8);
            showAlert("Location updated successfully", "success", "yield-alert");
        }, (error) => {
            console.warn("Geolocation error:", error);
            showAlert("Could not automatically fetching location. Added defaults.", "info", "yield-alert");
        });
    } else {
        showAlert("Geolocation not supported by this browser.", "error", "yield-alert");
    }
}

async function runYieldPrediction() {
    console.log('[Yield] runYieldPrediction called, file:', currentYieldFile);
    if (!currentYieldFile) {
        console.warn('[Yield] No file selected');
        return;
    }

    const btn = document.getElementById('yield-predict-btn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');

    // Loading state
    btn.disabled = true;
    btnText.textContent = "Analyzing Features...";
    spinner.classList.remove('hidden');
    hideAlert('yield-alert');

    const formData = new FormData();
    formData.append('file', currentYieldFile);
    
    // Add location data
    const lat = document.getElementById('yield-lat').value || 20.0;
    const lon = document.getElementById('yield-lon').value || 75.0;
    formData.append('lat', lat);
    formData.append('lon', lon);

    // Add economics data
    const pricePerKg = document.getElementById('yield-price').value || 200;
    const costPerPlant = document.getElementById('yield-cost-plant').value || 100;
    formData.append('price_per_kg', pricePerKg);
    formData.append('estimated_plant_maintenance_cost', costPerPlant);

    try {
        console.log('[Yield] Sending request to', `${API_URL}/predict/yield`);
        const response = await fetch(`${API_URL}/predict/yield`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorMsg = `Server error (${response.status})`;
            try {
                const error = await response.json();
                errorMsg = error.detail || errorMsg;
            } catch(e) {}
            throw new Error(errorMsg);
        }

        const data = await response.json();
        console.log('[Yield] Response data:', data);

        lastYieldData = data;

        // Check QA
        if (!data.ok) {
            document.getElementById('yield-result').classList.add('hidden');
            document.getElementById('yield-empty-state').style.display = 'flex';
            showAlert(data.message || "Yield Prediction Failed. No Coffee Fruit Found", "error", "yield-alert");
            if (data.qa) console.log('[Qa Failed]', data.qa);
            return;
        } else if (data.qa && data.qa.message) {
            showAlert(`✓ ${data.message || data.qa.message || "QA Passed"}`, "success", "yield-alert");
        } else {
            showAlert("Prediction Complete", "success", "yield-alert");
        }

        // Safe defaults — support both old and new API field names
        const predictedYield  = data.predicted_yield ?? 0;
        const fruitCount      = data.fruit_count ?? 0;
        const ripeCount       = data.ripe_count ?? data.ripe ?? 0;
        const unripeCount     = data.unripe_count ?? data.unripe ?? 0;
        const overripeCount   = data.overripe_count ?? data.overripe ?? 0;
        const ripenessRatio   = data.ripeness_ratio ?? 0.0;
        const temp            = data.temperature ?? 0.0;
        const hum             = data.humidity ?? 0;
        const rainToday       = data.rainfall_today ?? data.rainfall ?? 0.0;
        const rain30          = data.rainfall_last_30_days ?? data.rainfall ?? 0.0;
        const rain90          = data.rainfall_last_90_days ?? 0.0;
        const ripeningDays    = data.ripening_days ?? 0;
        const harvestDate     = data.expected_harvest_date ?? '--';

        const confidence    = data.confidence ?? 0.85;
        const modelArch     = data.model_architecture || 'Enhanced XGBoost (11-Feature)';
        const importances   = data.feature_importances || {};

        // New fields
        const profitStatus    = data.profit_status || 'LOSS';
        const profitAmount    = data.profit ?? 0;
        const breakEvenYield  = data.break_even_yield ?? 0;
        const explainerCauses         = data.explainer_causes || [];
        const explainerInsights       = data.explainer_insights || [];
        const explainerRecommendations= data.explainer_recommendations || [];
        const potentialYield  = data.potential_yield ?? predictedYield;
        const improvementPct  = data.improvement_pct ?? 0;

        // Soil fields
        const soilScore       = data.soil_score ?? 0.7;
        const soilTypeVal     = data.soil_type || 'Loamy';
        const waterRetVal     = data.water_retention || 'Medium';
        const drainageVal     = data.drainage || 'Moderate';

        // 1. Show Advanced Dashboard, hide empty state
        const yieldDashboard = document.getElementById('yield-result');
        yieldDashboard.classList.remove('hidden');
        const emptyState = document.getElementById('yield-empty-state');
        if (emptyState) emptyState.style.display = 'none';

        // 2. Populate Metrics Summary
        document.getElementById('yield-res-final').textContent = predictedYield.toFixed(1);
        document.getElementById('yield-res-model').textContent = modelArch;

        // 3. Profit Status Badge
        const profitBadge = document.getElementById('yield-profit-badge');
        const profitIcon = document.getElementById('yield-profit-icon');
        const profitLabel = document.getElementById('yield-profit-status');
        const profitAmountEl = document.getElementById('yield-profit-amount');

        profitBadge.className = 'profit-status-badge';
        if (profitStatus === 'PROFIT') {
            profitBadge.classList.add('profit-profit');
            profitIcon.textContent = '📈';
            profitLabel.textContent = 'PROFIT';
            profitAmountEl.style.color = 'var(--success)';
            profitAmountEl.textContent = `+₹${Math.abs(profitAmount).toLocaleString()}`;
        } else if (profitStatus === 'BREAK-EVEN') {
            profitBadge.classList.add('profit-breakeven');
            profitIcon.textContent = '⚖️';
            profitLabel.textContent = 'BREAK-EVEN';
            profitAmountEl.style.color = 'var(--warning)';
            profitAmountEl.textContent = '₹0';
        } else {
            profitBadge.classList.add('profit-loss');
            profitIcon.textContent = '📉';
            profitLabel.textContent = 'LOSS';
            profitAmountEl.style.color = 'var(--error)';
            profitAmountEl.textContent = `-₹${Math.abs(profitAmount).toLocaleString()}`;
        }

        // 4. Break-even & Profit cards
        document.getElementById('yield-break-even').textContent = breakEvenYield.toFixed(0);
        document.getElementById('yield-predicted-val').textContent = predictedYield.toFixed(0);
        document.getElementById('yield-potential-val').textContent = potentialYield.toFixed(0);
        document.getElementById('yield-improvement-pct').textContent = `↑ ${improvementPct}% possible`;

        const profitValEl = document.getElementById('yield-profit-val');
        if (profitAmount >= 0) {
            profitValEl.style.color = 'var(--success)';
            profitValEl.textContent = `+₹${Math.abs(profitAmount).toLocaleString()}`;
        } else {
            profitValEl.style.color = 'var(--error)';
            profitValEl.textContent = `-₹${Math.abs(profitAmount).toLocaleString()}`;
        }

        // 5. Populate Detailed Feature Cards
        document.getElementById('yield-res-count').textContent   = fruitCount;
        document.getElementById('yield-res-ripe').textContent    = ripeCount;
        document.getElementById('yield-res-unripe').textContent  = unripeCount;
        document.getElementById('yield-res-overripe').textContent = overripeCount;
        document.getElementById('yield-res-ratio').textContent   = ripenessRatio.toFixed(2);
        document.getElementById('yield-res-temp').textContent    = temp.toFixed(1) + '°C';
        document.getElementById('yield-res-hum').textContent     = Math.round(hum) + '%';
        document.getElementById('yield-res-rain-today').textContent = rainToday.toFixed(1) + 'mm';
        document.getElementById('yield-res-rain').textContent   = rain30.toFixed(1) + 'mm';
        document.getElementById('yield-res-rain90').textContent = rain90.toFixed(1) + 'mm';

        // Ripening time cards
        const ripeningEl = document.getElementById('yield-res-ripening-days');
        if (ripeningDays > 0) {
            ripeningEl.textContent = ripeningDays + ' days';
            ripeningEl.style.color = ripeningDays <= 35 ? '#3fb950'
                : ripeningDays <= 40 ? '#d29922' : '#f85149';
        } else {
            ripeningEl.textContent = 'Ready';
            ripeningEl.style.color = '#3fb950';
        }
        document.getElementById('yield-res-harvest-date').textContent = harvestDate;

        // 5.5 Soil Score display
        document.getElementById('yield-res-soil-score').textContent = soilScore.toFixed(2);
        document.getElementById('yield-res-soil-type').textContent = soilTypeVal.charAt(0).toUpperCase() + soilTypeVal.slice(1);
        document.getElementById('yield-res-water-ret').textContent = waterRetVal.charAt(0).toUpperCase() + waterRetVal.slice(1);
        document.getElementById('yield-res-drainage').textContent = drainageVal.charAt(0).toUpperCase() + drainageVal.slice(1);
        // Animate soil score bar
        setTimeout(() => {
            document.getElementById('yield-soil-bar-fill').style.width = `${Math.round(soilScore * 100)}%`;
        }, 200);

        // 5.6 Display SoilGrids Detail optionally
        const yieldSoilGridsDisplay = document.getElementById('yield-soil-grids-display');
        if (data.soil_grids_data) {
            yieldSoilGridsDisplay.classList.remove('hidden');
            document.getElementById('yield-res-soil-ph').textContent = data.soil_grids_data.ph ? data.soil_grids_data.ph.toFixed(1) : 'N/A';
            document.getElementById('yield-res-soil-clay').textContent = data.soil_grids_data.clay ? data.soil_grids_data.clay.toFixed(1) + ' %' : 'N/A';
            document.getElementById('yield-res-soil-sand').textContent = data.soil_grids_data.sand ? data.soil_grids_data.sand.toFixed(1) + ' %' : 'N/A';
            document.getElementById('yield-res-soil-silt').textContent = data.soil_grids_data.silt ? data.soil_grids_data.silt.toFixed(1) + ' %' : 'N/A';
            document.getElementById('yield-res-soil-soc').textContent = data.soil_grids_data.soc ? data.soil_grids_data.soc.toFixed(1) + ' g/kg' : 'N/A';
        } else {
            yieldSoilGridsDisplay.classList.add('hidden');
        }
        
        // 6. Update Visualization
        document.getElementById('yield-det-count-badge').textContent = `${fruitCount} Detected`;
        if (data.annotated_image) {
            document.getElementById('yield-result-img').src = `data:image/jpeg;base64,${data.annotated_image}`;
        } else {
            const previewImg = document.getElementById('yield-preview-img');
            if (previewImg) document.getElementById('yield-result-img').src = previewImg.src;
        }

        // 7. Animate Confidence Bar
        const confValue = Math.round(confidence * 100);
        document.getElementById('yield-res-conf').textContent = `${confValue}%`;
        setTimeout(() => {
            document.getElementById('yield-conf-fill').style.width = `${confValue}%`;
        }, 100);

        // 8. Render/Update Feature Importance Chart
        renderYieldImportanceChart(importances);

        // 9. Render Explainability Sections
        
        // Soil Summary
        const soilSummaryContainer = document.getElementById('yield-soil-summary');
        if (soilSummaryContainer) {
            let summaryText = `Detected <strong>${soilTypeVal}</strong> soil with <strong>${drainageVal}</strong> drainage and <strong>${waterRetVal}</strong> water retention capabilities.`;
            if (data.soil_grids_data && data.soil_grids_data.ph) {
                summaryText += `<br>SoilGrids Analysis indicates pH <strong>${data.soil_grids_data.ph.toFixed(1)}</strong> and SOC <strong>${data.soil_grids_data.soc.toFixed(1)} g/kg</strong>.`;
            }
            soilSummaryContainer.innerHTML = summaryText;
        }

        // Dynamic Explainability
        const dynamic = data.dynamic_insights || {};
        
        // Primary Stress
        const primaryContainer = document.getElementById('yield-primary-stress');
        if (primaryContainer && dynamic.primary_stress) {
            const sevColor = dynamic.primary_stress.severity === 'high' ? '#f85149' : (dynamic.primary_stress.severity === 'medium' ? '#ffab40' : '#d29922');
            primaryContainer.innerHTML = `
                <div style="margin-bottom: 8px;">
                    <span style="background: ${sevColor}22; color: ${sevColor}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase;">${dynamic.primary_stress.severity}</span>
                    <strong style="margin-left: 8px; font-size: 13px;">${dynamic.primary_stress.name}</strong>
                </div>
                <p style="margin: 0;">${dynamic.primary_stress.description}</p>
            `;
        }
        
        // Secondary Stress
        const secondaryContainer = document.getElementById('yield-secondary-stress');
        if (secondaryContainer && dynamic.secondary_stress) {
            const sevColor = dynamic.secondary_stress.severity === 'high' ? '#f85149' : (dynamic.secondary_stress.severity === 'medium' ? '#ffab40' : '#d29922');
            secondaryContainer.innerHTML = `
                <div style="margin-bottom: 8px;">
                    <span style="background: ${sevColor}22; color: ${sevColor}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase;">${dynamic.secondary_stress.severity}</span>
                    <strong style="margin-left: 8px; font-size: 13px;">${dynamic.secondary_stress.name}</strong>
                </div>
                <p style="margin: 0;">${dynamic.secondary_stress.description}</p>
            `;
        }

        // Crop Insights
        const cropInsightsContainer = document.getElementById('yield-crop-insights');
        if (cropInsightsContainer && dynamic.crop_insights) {
            cropInsightsContainer.innerHTML = '';
            dynamic.crop_insights.forEach(insight => {
                const li = document.createElement('li');
                li.style.marginBottom = '6px';
                li.textContent = insight;
                cropInsightsContainer.appendChild(li);
            });
        }
        
        // Root Insights
        const rootInsightsContainer = document.getElementById('yield-root-insights');
        if (rootInsightsContainer && dynamic.root_insights) {
            rootInsightsContainer.innerHTML = '';
            dynamic.root_insights.forEach(insight => {
                const li = document.createElement('li');
                li.style.marginBottom = '6px';
                li.textContent = insight;
                rootInsightsContainer.appendChild(li);
            });
        }
        
        // Recommendations
        const recsContainer = document.getElementById('yield-dynamic-recs');
        if (recsContainer && dynamic.recommendations) {
            recsContainer.innerHTML = '';
            dynamic.recommendations.forEach(rec => {
                const color = rec.priority === 'high' ? '#f85149' : (rec.priority === 'medium' ? '#ffab40' : '#3fb950');
                const el = document.createElement('div');
                el.style.cssText = `padding:10px 12px; border-radius:6px; background:rgba(255,255,255,0.05); border-left:3px solid ${color};`;
                el.innerHTML = `
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                        <strong style="font-size:12px;color:var(--text-primary);">${rec.action}</strong>
                        <span style="margin-left:auto;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700;
                                     background:${color}22;color:${color};text-transform:uppercase;">${rec.priority}</span>
                    </div>
                    <p style="font-size:11px;color:var(--text-muted);margin:0;">${rec.reason}</p>
                `;
                recsContainer.appendChild(el);
            });
        }

        showAlert('✓ Intelligent yield analysis completed successfully', 'success', 'yield-alert');
        
        // Scroll to dashboard
        yieldDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        showAlert(`Yield analysis failed: ${error.message}`, 'error', 'yield-alert');
        console.error('[Yield] Error:', error);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Predict Yield';
        spinner.classList.add('hidden');
    }
}

function renderYieldImportanceChart(importances) {
    const ctx = document.getElementById('yieldImportanceChart').getContext('2d');
    
    const labels = Object.keys(importances);
    const values = Object.values(importances);

    if (yieldChart) {
        yieldChart.destroy();
    }

    yieldChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Contribution',
                data: values,
                backgroundColor: 'rgba(198, 124, 78, 0.4)',
                borderColor: '#c67c4e',
                borderWidth: 2,
                borderRadius: 4,
                hoverBackgroundColor: 'rgba(198, 124, 78, 0.7)'
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(13, 17, 23, 0.9)',
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#484f58', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#8b949e', font: { size: 11 } }
                }
            }
        }
    });
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

async function downloadYieldPDFReport(event) {
    event.preventDefault(); // Prevent default anchor jump

    if (!lastYieldData) return;

    // Ensure original image is included for report fallback
    const reportData = {
        ...lastYieldData,
        original_image: yieldOriginalImageBase64
    };

    const btn = document.getElementById('yield-download-btn');
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
        a.download = `CoffeeAI_Yield_Report_${new Date().getTime()}.pdf`;
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
    document.getElementById('yield-result').classList.add('hidden');
    document.getElementById('result-img').src = ""; // Clear to prevent ghosting
    document.querySelector('.distribution-section').style.display = 'block'; // Restore for maturity
}



// ==================== DASHBOARD ====================
async function loadDashboard() {
    try {
        const response = await fetch(`${API_URL}/dashboard-stats`, { cache: 'no-store' });
        if (!response.ok) return;

        const data = await response.json();
        const mat = data.maturity || {};
        const yld = data.yield    || {};

        // ---- MATURITY CARDS ----
        document.getElementById('dash-maturity-total').textContent = mat.total_predictions ?? 0;

        document.getElementById('dash-yolo').textContent   = mat.yolo_avg > 0
            ? (mat.yolo_avg * 100).toFixed(1) + '%' : '--';
        document.getElementById('dash-effnet').textContent = mat.efficientnet_avg > 0
            ? (mat.efficientnet_avg * 100).toFixed(1) + '%' : '--';
        document.getElementById('dash-hybrid').textContent = mat.hybrid_avg > 0
            ? (mat.hybrid_avg * 100).toFixed(1) + '%' : '--';

        // ---- YIELD CARDS ----
        document.getElementById('dash-yield-total').textContent = yld.total_yield_predictions ?? 0;

        document.getElementById('dash-yield-avg').textContent   = yld.avg_yield > 0
            ? yld.avg_yield.toFixed(1) + ' kg/ha' : '--';
        document.getElementById('dash-yield-ratio').textContent = yld.avg_ripeness > 0
            ? yld.avg_ripeness.toFixed(2) : '--';
        document.getElementById('dash-yield-count').textContent = yld.avg_fruit_count > 0
            ? Math.round(yld.avg_fruit_count) : '--';

        const ripeningCard = document.getElementById('dash-yield-ripening');
        if (ripeningCard) {
            ripeningCard.textContent = yld.avg_ripening_days > 0
                ? Math.round(yld.avg_ripening_days) + ' days' : '-- days';
        }

        // Soil Score
        const soilCard = document.getElementById('dash-yield-soil');
        if (soilCard) {
            soilCard.textContent = yld.avg_soil_score > 0
                ? yld.avg_soil_score.toFixed(2) : '--';
        }

    } catch (e) {
        console.warn("Dashboard load failed:", e);
    }
}


function showExportDialog() {
    const dialog = document.getElementById('export-dialog');
    dialog.classList.remove('hidden');
    dialog.style.display = 'flex';
}

function hideExportDialog() {
    const dialog = document.getElementById('export-dialog');
    dialog.classList.add('hidden');
    dialog.style.display = 'none';
}

function handleCSVExport(e) {
    e.preventDefault();
    const type = document.querySelector('input[name="export-type"]:checked').value;
    window.location.href = `${API_URL}/data/export/csv?type=${type}`;
    hideExportDialog();
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
loadDashboard();


// ============================================================
// ==================== COMPARE PLANTS ========================
// ============================================================

let comparePlantAFile = null;
let comparePlantBFile = null;
let compareYieldChartInstance = null;
let compareSoilChartInstance  = null;

// ── File Handling ──────────────────────────────────────────────
function handleComparePlantFile(plant, files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (plant === 'A') {
        comparePlantAFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            document.getElementById('compare-preview-a').src = e.target.result;
            document.getElementById('compare-placeholder-a').classList.add('hidden');
            document.getElementById('compare-preview-container-a').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        comparePlantBFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            document.getElementById('compare-preview-b').src = e.target.result;
            document.getElementById('compare-placeholder-b').classList.add('hidden');
            document.getElementById('compare-preview-container-b').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
    _updateCompareRunBtn();
}

function clearComparePlant(plant, event) {
    if (event) { event.stopPropagation(); event.preventDefault(); }
    if (plant === 'A') {
        comparePlantAFile = null;
        document.getElementById('compare-preview-a').src = '';
        document.getElementById('compare-preview-container-a').classList.add('hidden');
        document.getElementById('compare-placeholder-a').classList.remove('hidden');
        document.getElementById('compareFileA').value = '';
    } else {
        comparePlantBFile = null;
        document.getElementById('compare-preview-b').src = '';
        document.getElementById('compare-preview-container-b').classList.add('hidden');
        document.getElementById('compare-placeholder-b').classList.remove('hidden');
        document.getElementById('compareFileB').value = '';
    }
    _updateCompareRunBtn();
}

function _updateCompareRunBtn() {
    document.getElementById('compare-run-btn').disabled = !(comparePlantAFile && comparePlantBFile);
}

// ── GPS Helpers ────────────────────────────────────────────────
function getComparePlantLocation(plant) {
    if (!navigator.geolocation) {
        showAlert('Geolocation not supported by this browser.', 'error', 'compare-alert');
        return;
    }
    navigator.geolocation.getCurrentPosition(
        pos => {
            document.getElementById(`compare-lat-${plant.toLowerCase()}`).value = pos.coords.latitude.toFixed(4);
            document.getElementById(`compare-lon-${plant.toLowerCase()}`).value = pos.coords.longitude.toFixed(4);
            showAlert(`GPS location set for Plant ${plant}`, 'success', 'compare-alert');
        },
        err => {
            console.warn('Geolocation error:', err);
            showAlert(`Could not get location for Plant ${plant}. Using default.`, 'info', 'compare-alert');
        }
    );
}

// ── Main Prediction ─────────────────────────────────────────────
async function runComparePrediction() {
    if (!comparePlantAFile || !comparePlantBFile) return;

    const btn     = document.getElementById('compare-run-btn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');

    btn.disabled = true;
    btnText.textContent = 'Analysing Both Plants...';
    spinner.classList.remove('hidden');
    hideAlert('compare-alert');
    document.getElementById('compare-results').classList.add('hidden');

    const formData = new FormData();
    formData.append('file_a', comparePlantAFile);
    formData.append('file_b', comparePlantBFile);
    formData.append('lat_a', document.getElementById('compare-lat-a').value || 13.0);
    formData.append('lon_a', document.getElementById('compare-lon-a').value || 76.1);
    formData.append('lat_b', document.getElementById('compare-lat-b').value || 13.0);
    formData.append('lon_b', document.getElementById('compare-lon-b').value || 76.1);
    formData.append('price_per_kg',                       document.getElementById('compare-price').value || 200);
    formData.append('estimated_plant_maintenance_cost',   document.getElementById('compare-cost').value  || 100);

    try {
        const response = await fetch(`${API_URL}/predict/compare`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let msg = `Server error (${response.status})`;
            try { const e = await response.json(); msg = e.detail || msg; } catch(_){}
            throw new Error(msg);
        }

        const data = await response.json();
        console.log('[Compare] Response:', data);

        displayCompareResults(data);
        showAlert('✓ Comparative analysis completed successfully', 'success', 'compare-alert');

    } catch (err) {
        console.error('[Compare] Error:', err);
        showAlert(`Error: ${err.message}`, 'error', 'compare-alert');
    } finally {
        btn.disabled = false;
        btnText.textContent = '⚡ Compare Plants';
        spinner.classList.add('hidden');
        _updateCompareRunBtn();
    }
}

// ── Display Results ─────────────────────────────────────────────
function displayCompareResults(data) {
    const { plant_a, plant_b, comparison } = data;
    const yc = comparison.yield_comparison || {};
    const sameLocation = comparison.same_location_mode === true;

    // ── Same-location mode banner ────────────────────────────────
    let sameBanner = document.getElementById('compare-same-loc-banner');
    if (!sameBanner) {
        sameBanner = document.createElement('div');
        sameBanner.id = 'compare-same-loc-banner';
        document.getElementById('compare-verdict-banner').insertAdjacentElement('afterend', sameBanner);
    }

    if (sameLocation) {
        sameBanner.style.cssText = `
            display:flex;align-items:flex-start;gap:12px;padding:12px 16px;border-radius:10px;
            margin-bottom:16px;background:rgba(137,87,229,0.08);border:1px solid rgba(137,87,229,0.3);`;
        sameBanner.innerHTML = `
            <span style="font-size:20px;flex-shrink:0;">📡</span>
            <div>
                <strong style="font-size:12px;color:#8957e5;display:block;margin-bottom:4px;">
                    Shared Environmental Context Detected — Visual Comparative Analysis Mode
                </strong>
                <p style="margin:0;font-size:11px;color:var(--text-secondary);line-height:1.5;">
                    ${comparison.shared_context_note || 'Both plants share the same soil and environmental conditions. Analysis focuses on visual plant-level differences.'}
                </p>
            </div>`;
    } else {
        sameBanner.style.display = 'none';
    }

    // ── Verdict banner ───────────────────────────────────────────
    document.getElementById('compare-verdict-text').textContent = comparison.summary || comparison.verdict || '';

    // ── Plant summaries ──────────────────────────────────────────
    // Pass winner key so badges are always High vs Low (relative, not absolute)
    const winner = yc.winner; // 'A' or 'B'
    _fillPlantSummary('a', plant_a, '#62c170', winner === 'A');
    _fillPlantSummary('b', plant_b, '#ffab40', winner === 'B');

    // Highlight loser
    const loser = yc.loser;
    if (loser) {
        document.getElementById(`compare-card-${loser.toLowerCase()}`).style.outline = '2px solid rgba(248,81,73,0.5)';
    }
    document.getElementById('compare-loser-badge').textContent = loser ? `⚠️ Lower: Plant ${loser}` : '';

    // Annotated thumbnails
    if (plant_a.annotated_image)
        document.getElementById('compare-thumb-a').src = `data:image/jpeg;base64,${plant_a.annotated_image}`;
    if (plant_b.annotated_image)
        document.getElementById('compare-thumb-b').src = `data:image/jpeg;base64,${plant_b.annotated_image}`;

    // ── Visual Scores Panel (same-location only) ─────────────────
    _renderVisualScoresPanel(comparison, sameLocation);

    // ── Charts ───────────────────────────────────────────────────
    renderCompareYieldChart(yc);
    if (sameLocation && comparison.visual_scores_a && comparison.visual_scores_b) {
        renderCompareVisualChart(comparison.visual_scores_a, comparison.visual_scores_b);
    } else {
        renderCompareSoilChart(plant_a.soil_grids_data || {}, plant_b.soil_grids_data || {});
    }

    // ── Comparison table ─────────────────────────────────────────
    _buildCompareTable(plant_a, plant_b, comparison.soil_comparison || [], sameLocation);

    // ── Dynamic Explainability ──────────────────────────────────────────────
    const dynamic = comparison.dynamic_insights || {};

    // Primary Stress
    const primaryContainer = document.getElementById('compare-primary-stress');
    if (primaryContainer && dynamic.primary_stress) {
        const sevColor = dynamic.primary_stress.severity === 'high' ? '#f85149' : (dynamic.primary_stress.severity === 'medium' ? '#ffab40' : '#d29922');
        primaryContainer.innerHTML = `
            <div style="margin-bottom: 8px;">
                <span style="background: ${sevColor}22; color: ${sevColor}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase;">${dynamic.primary_stress.severity}</span>
                <strong style="margin-left: 8px; font-size: 13px;">${dynamic.primary_stress.name}</strong>
            </div>
            <p style="margin: 0;">${dynamic.primary_stress.description}</p>
        `;
    } else if (primaryContainer) {
        primaryContainer.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No major stress identified.</p>';
    }

    // Secondary Stress
    const secondaryContainer = document.getElementById('compare-secondary-stress');
    if (secondaryContainer && dynamic.secondary_stress) {
        const sevColor = dynamic.secondary_stress.severity === 'high' ? '#f85149' : (dynamic.secondary_stress.severity === 'medium' ? '#ffab40' : '#d29922');
        secondaryContainer.innerHTML = `
            <div style="margin-bottom: 8px;">
                <span style="background: ${sevColor}22; color: ${sevColor}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase;">${dynamic.secondary_stress.severity}</span>
                <strong style="margin-left: 8px; font-size: 13px;">${dynamic.secondary_stress.name}</strong>
            </div>
            <p style="margin: 0;">${dynamic.secondary_stress.description}</p>
        `;
    } else if (secondaryContainer) {
        secondaryContainer.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No secondary stress identified.</p>';
    }

    // Crop Insights
    const cropInsightsContainer = document.getElementById('compare-crop-insights');
    if (cropInsightsContainer) {
        cropInsightsContainer.innerHTML = '';
        (dynamic.crop_insights || []).forEach(insight => {
            const li = document.createElement('li');
            li.style.marginBottom = '6px';
            li.textContent = insight;
            cropInsightsContainer.appendChild(li);
        });
        if (!dynamic.crop_insights || dynamic.crop_insights.length === 0) {
            cropInsightsContainer.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No specific crop health insights.</p>';
        }
    }

    // Root Insights
    const rootInsightsContainer = document.getElementById('compare-root-insights');
    if (rootInsightsContainer) {
        rootInsightsContainer.innerHTML = '';
        (dynamic.root_insights || []).forEach(insight => {
            const li = document.createElement('li');
            li.style.marginBottom = '6px';
            li.textContent = insight;
            rootInsightsContainer.appendChild(li);
        });
        if (!dynamic.root_insights || dynamic.root_insights.length === 0) {
            rootInsightsContainer.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No specific root-zone insights.</p>';
        }
    }

    // Recommendations
    const recsContainer = document.getElementById('compare-dynamic-recs');
    if (recsContainer) {
        recsContainer.innerHTML = '';
        (dynamic.recommendations || []).forEach(rec => {
            const color = rec.priority === 'high' ? '#f85149' : (rec.priority === 'medium' ? '#ffab40' : '#3fb950');
            const el = document.createElement('div');
            el.style.cssText = `padding:10px 12px; border-radius:6px; background:rgba(255,255,255,0.05); border-left:3px solid ${color};`;
            el.innerHTML = `
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    <strong style="font-size:12px;color:var(--text-primary);">${rec.action}</strong>
                    <span style="margin-left:auto;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700;
                                 background:${color}22;color:${color};text-transform:uppercase;">${rec.priority}</span>
                </div>
                <p style="font-size:11px;color:var(--text-muted);margin:0;">${rec.reason}</p>
            `;
            recsContainer.appendChild(el);
        });
        if (!dynamic.recommendations || dynamic.recommendations.length === 0) {
            recsContainer.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No specific recommendations needed.</p>';
        }
    }

    // Show results
    document.getElementById('compare-results').classList.remove('hidden');
    document.getElementById('compare-page').scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Visual Scores Panel (same-location mode) ──────────────────────
function _renderVisualScoresPanel(comparison, sameLocation) {
    let panel = document.getElementById('compare-visual-scores-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'compare-visual-scores-panel';
        // Insert before the charts row
        const chartsRow = document.querySelector('#compare-results [style*="compareYieldChart"]')?.closest('[style*="grid"]');
        if (chartsRow) chartsRow.insertAdjacentElement('beforebegin', panel);
    }

    const va = comparison.visual_scores_a;
    const vb = comparison.visual_scores_b;

    if (!sameLocation || !va || !vb) {
        panel.style.display = 'none';
        return;
    }

    panel.style.cssText = 'margin-bottom:16px;';
    panel.innerHTML = `
        <div class="glass-card" style="padding:18px;border-left:3px solid #8957e5;">
            <h4 style="font-size:13px;color:#8957e5;margin-bottom:14px;">🔬 Visual Plant Analysis — Shared Location Context</h4>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
                ${_visScoreCard('Fruit Density', va.density_score, vb.density_score, v => (v*100).toFixed(0)+'%')}
                ${_visScoreCard('Ripeness Ratio', va.ripeness_score, vb.ripeness_score, v => (v*100).toFixed(0)+'%')}
                ${_visScoreCard('Maturity Consistency', va.consistency_score, vb.consistency_score, v => (v*100).toFixed(0)+'%')}
                ${_visScoreCard('Ripe : Unripe Ratio', va.ripe_unripe_ratio, vb.ripe_unripe_ratio, v => v.toFixed(2))}
            </div>
        </div>`;
}

function _visScoreCard(label, va, vb, fmt) {
    const aWins = va >= vb;
    const colA  = aWins ? '#62c170' : '#f85149';
    const colB  = !aWins ? '#62c170' : '#f85149';
    return `
        <div style="background:rgba(0,0,0,0.2);padding:10px;border-radius:8px;text-align:center;">
            <div style="font-size:9px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;">${label}</div>
            <div style="display:flex;justify-content:space-around;align-items:center;">
                <div><div style="font-size:9px;color:#62c170;margin-bottom:2px;">Plant A</div>
                     <strong style="font-size:13px;color:${colA};">${fmt(va)}</strong></div>
                <div style="color:var(--text-muted);font-size:12px;">vs</div>
                <div><div style="font-size:9px;color:#ffab40;margin-bottom:2px;">Plant B</div>
                     <strong style="font-size:13px;color:${colB};">${fmt(vb)}</strong></div>
            </div>
        </div>`;
}


function _fillPlantSummary(suffix, plant, color, isWinner) {
    const s = suffix;
    const yield_kg = plant.predicted_yield ?? 0;

    // Always show High Yield for winner, Low Yield for loser
    // so both cards never show the same category
    let category, catColor;
    if (isWinner === undefined) {
        // Fallback: absolute threshold (used outside compare context)
        category = yield_kg > 1.5 ? 'High' : yield_kg > 0.8 ? 'Medium' : 'Low';
        catColor  = yield_kg > 1.5 ? '#3fb950' : yield_kg > 0.8 ? '#d29922' : '#f85149';
    } else if (isWinner) {
        category = 'High Yield';
        catColor  = '#3fb950';  // green
    } else {
        category = 'Low Yield';
        catColor  = '#f85149';  // red
    }

    document.getElementById(`compare-badge-${s}`).textContent   = category;
    document.getElementById(`compare-badge-${s}`).style.color   = catColor;
    document.getElementById(`compare-badge-${s}`).style.background = `${catColor}22`;
    document.getElementById(`cmp-yield-${s}`).textContent  = `${yield_kg.toFixed(3)} kg`;
    document.getElementById(`cmp-fruit-${s}`).textContent  = plant.fruit_count ?? '--';
    document.getElementById(`cmp-ripe-${s}`).textContent   = plant.ripeness_ratio != null ? `${(plant.ripeness_ratio * 100).toFixed(0)}%` : '--';
    document.getElementById(`cmp-soil-${s}`).textContent   = plant.soil_score != null ? plant.soil_score.toFixed(2) : '--';
}

function _buildCompareTable(a, b, soilCmp, sameLocation) {
    const tbody = document.getElementById('compare-table-body');
    tbody.innerHTML = '';

    const fixed = [
        { label: '🌾 Predicted Yield (kg/plant)', va: a.predicted_yield?.toFixed(3), vb: b.predicted_yield?.toFixed(3), adv: (a.predicted_yield ?? 0) >= (b.predicted_yield ?? 0) ? 'A' : 'B' },
        { label: '🍒 Fruit Count',                va: a.fruit_count,                 vb: b.fruit_count,                 adv: (a.fruit_count ?? 0) >= (b.fruit_count ?? 0) ? 'A' : 'B' },
        { label: '✅ Ripeness Ratio',              va: a.ripeness_ratio != null ? `${(a.ripeness_ratio*100).toFixed(0)}%` : '--',
                                                   vb: b.ripeness_ratio != null ? `${(b.ripeness_ratio*100).toFixed(0)}%` : '--',
                                                   adv: (a.ripeness_ratio ?? 0) >= (b.ripeness_ratio ?? 0) ? 'A' : 'B' },
        { label: '🌡️ Temperature (°C)',            va: a.temperature?.toFixed(1),     vb: b.temperature?.toFixed(1),     adv: 'equal' },
        { label: '💧 Humidity (%)',                va: a.humidity?.toFixed(0),        vb: b.humidity?.toFixed(0),        adv: 'equal' },
        { label: '🌧️ Rainfall 30d (mm)',           va: a.rainfall_last_30_days?.toFixed(1), vb: b.rainfall_last_30_days?.toFixed(1), adv: (a.rainfall_last_30_days ?? 0) >= (b.rainfall_last_30_days ?? 0) ? 'A' : 'B' },
        { label: '🌱 Soil Score',                  va: a.soil_score?.toFixed(2),      vb: b.soil_score?.toFixed(2),      adv: (a.soil_score ?? 0) >= (b.soil_score ?? 0) ? 'A' : 'B' },
    ];

    // In same-location mode add visual metric rows; otherwise add soil rows
    if (sameLocation) {
        fixed.push(
            { label: '🔬 Ripe Count',        va: a.ripe_count,    vb: b.ripe_count,    adv: (a.ripe_count ?? 0) >= (b.ripe_count ?? 0) ? 'A' : 'B' },
            { label: '🟡 Unripe Count',      va: a.unripe_count,  vb: b.unripe_count,  adv: (a.unripe_count ?? 0) <= (b.unripe_count ?? 0) ? 'A' : 'B' },
            { label: '🔴 Overripe Count',    va: a.overripe_count, vb: b.overripe_count, adv: (a.overripe_count ?? 0) <= (b.overripe_count ?? 0) ? 'A' : 'B' }
        );
    } else {
        soilCmp.forEach(row => fixed.push({ label: row.label, va: row.value_a ?? 'N/A', vb: row.value_b ?? 'N/A', adv: row.advantage }));
    }

    fixed.forEach(row => {
        const tr = document.createElement('tr');
        const advCell = row.adv === 'A' ? '<span style="color:#62c170;">🌿 A</span>'
                      : row.adv === 'B' ? '<span style="color:#ffab40;">🌿 B</span>'
                      : '<span style="color:var(--text-muted);">—</span>';
        tr.innerHTML = `
            <td style="padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.04);color:var(--text-secondary);">${row.label}</td>
            <td style="padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.04);color:#62c170;font-weight:600;">${row.va ?? '--'}</td>
            <td style="padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.04);color:#ffab40;font-weight:600;">${row.vb ?? '--'}</td>
            <td style="padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.04);">${advCell}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ── Charts ──────────────────────────────────────────────────────
// Visual fruit distribution chart (same-location mode)
function renderCompareVisualChart(va, vb) {
    const ctx = document.getElementById('compareSoilChart').getContext('2d');
    if (compareSoilChartInstance) compareSoilChartInstance.destroy();
    compareSoilChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Fruit Count', 'Ripe', 'Unripe', 'Overripe'],
            datasets: [
                { label: 'Plant A', data: [va.fruit_count, va.ripe_count, va.unripe_count, va.overripe_count],
                  backgroundColor: 'rgba(98,193,112,0.55)', borderColor: '#62c170', borderWidth: 1.5, borderRadius: 4 },
                { label: 'Plant B', data: [vb.fruit_count, vb.ripe_count, vb.unripe_count, vb.overripe_count],
                  backgroundColor: 'rgba(255,171,64,0.55)',  borderColor: '#ffab40', borderWidth: 1.5, borderRadius: 4 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } },
            scales: {
                y: { beginAtZero: true, ticks: { color: '#8b949e', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#8b949e', font: { size: 9 } }, grid: { display: false } }
            }
        }
    });
}

function renderCompareYieldChart(yc) {
    const ctx = document.getElementById('compareYieldChart').getContext('2d');
    if (compareYieldChartInstance) compareYieldChartInstance.destroy();
    compareYieldChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Plant A', 'Plant B'],
            datasets: [{
                label: 'Predicted Yield (kg/plant)',
                data: [yc.yield_a ?? 0, yc.yield_b ?? 0],
                backgroundColor: ['rgba(98,193,112,0.6)', 'rgba(255,171,64,0.6)'],
                borderColor:     ['#62c170', '#ffab40'],
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => ` ${c.raw.toFixed(3)} kg/plant` } }
            },
            scales: {
                y: { beginAtZero: true, ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#8b949e', font: { size: 11 } }, grid: { display: false } }
            }
        }
    });
}

function renderCompareSoilChart(soilA, soilB) {
    const ctx = document.getElementById('compareSoilChart').getContext('2d');
    if (compareSoilChartInstance) compareSoilChartInstance.destroy();

    const labels = ['pH', 'SOC (g/kg)', 'Clay (%)', 'Sand (%)', 'Silt (%)'];
    const keysA  = [soilA.ph, soilA.soc, soilA.clay, soilA.sand, soilA.silt];
    const keysB  = [soilB.ph, soilB.soc, soilB.clay, soilB.sand, soilB.silt];

    compareSoilChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Plant A',
                    data: keysA.map(v => v ?? 0),
                    backgroundColor: 'rgba(98,193,112,0.55)',
                    borderColor: '#62c170',
                    borderWidth: 1.5,
                    borderRadius: 4,
                },
                {
                    label: 'Plant B',
                    data: keysB.map(v => v ?? 0),
                    backgroundColor: 'rgba(255,171,64,0.55)',
                    borderColor: '#ffab40',
                    borderWidth: 1.5,
                    borderRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#8b949e', font: { size: 10 } } },
            },
            scales: {
                y: { beginAtZero: true, ticks: { color: '#8b949e', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#8b949e', font: { size: 9 } }, grid: { display: false } }
            }
        }
    });
}

// ── Reset ──────────────────────────────────────────────────────
function resetComparePrediction() {
    clearComparePlant('A', null);
    clearComparePlant('B', null);
    document.getElementById('compare-results').classList.add('hidden');
    document.getElementById('compare-card-a').style.outline = '';
    document.getElementById('compare-card-b').style.outline = '';
    document.getElementById('compare-loser-badge').textContent = '';
    document.getElementById('compare-table-body').innerHTML = '';
    document.getElementById('compare-causes-list').innerHTML = '';
    document.getElementById('compare-health-list').innerHTML = '';
    document.getElementById('compare-recs-list').innerHTML = '';
    document.getElementById('compare-verdict-text').textContent = '';
    if (compareYieldChartInstance) { compareYieldChartInstance.destroy(); compareYieldChartInstance = null; }
    if (compareSoilChartInstance)  { compareSoilChartInstance.destroy();  compareSoilChartInstance  = null; }
    hideAlert('compare-alert');
    document.getElementById('compare-page').scrollTo({ top: 0, behavior: 'smooth' });
}
