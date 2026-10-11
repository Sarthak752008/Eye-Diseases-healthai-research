/* ==========================================================================
   EyeHealthAI — Research Dashboard Client Logic (Layers 5 & 6)
   Supports File Upload and Live Laptop Webcam Capture with Quality Guidance.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  let currentModule = 'anemia';
  let activeMode = 'upload'; // 'upload' or 'webcam'
  let uploadedFiles = []; // Array of { file, viewType, previewUrl, quality, dimensions }
  let lastSessionData = null;
  let currentSessionRequestId = 0; // Guard against race conditions and stale async responses

  // Camera State
  let mediaStream = null;
  let capturedWebcamBlob = null;
  let capturedWebcamUrl = null;
  let capturedQuality = null;

  // DOM Elements - Navigation & Cards
  const moduleCards = document.querySelectorAll('.module-card');
  const moduleBanner = document.getElementById('moduleBanner');
  const btnModeUpload = document.getElementById('btnModeUpload');
  const btnModeWebcam = document.getElementById('btnModeWebcam');
  const uploadSection = document.getElementById('uploadSection');
  const webcamSection = document.getElementById('webcamSection');

  // DOM Elements - Upload Mode
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewsContainer = document.getElementById('previewsContainer');
  const qualityWarningBox = document.getElementById('qualityWarningBox');

  // DOM Elements - Webcam Mode
  const webcamVideo = document.getElementById('webcamVideo');
  const webcamCanvas = document.getElementById('webcamCanvas');
  const btnStartCamera = document.getElementById('btnStartCamera');
  const btnCaptureStill = document.getElementById('btnCaptureStill');
  const btnStopCamera = document.getElementById('btnStopCamera');
  const btnRecapture = document.getElementById('btnRecapture');
  const webcamStatusAlert = document.getElementById('webcamStatusAlert');
  const capturedStillCard = document.getElementById('capturedStillCard');
  const capturedStillThumb = document.getElementById('capturedStillThumb');
  const webcamQualityMetrics = document.getElementById('webcamQualityMetrics');
  const webcamGuidanceBox = document.getElementById('webcamGuidanceBox');
  const webcamViewSelect = document.getElementById('webcamViewSelect');
  const chkWebcamOverride = document.getElementById('chkWebcamOverride');
  const webcamDomainNotice = document.getElementById('webcamDomainNotice');

  // DOM Elements - Results & Actions
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnAnalyzeText = document.getElementById('btnAnalyzeText');
  const btnSpinner = document.getElementById('btnSpinner');
  const resultsEmpty = document.getElementById('resultsEmpty');
  const resultsContent = document.getElementById('resultsContent');
  const btnResetSession = document.getElementById('btnResetSession');
  const btnExport = document.getElementById('btnExport');

  // View Options for Anemia
  const ANEMIA_VIEWS = [
    { value: 'palpebral', label: 'Palpebral Conjunctiva' },
    { value: 'forniceal', label: 'Forniceal Conjunctiva' },
    { value: 'forniceal_palpebral', label: 'Forniceal + Palpebral' },
    { value: 'raw_eye', label: 'Raw Full Eye View' },
    { value: 'unspecified', label: 'Unspecified Ocular Crop' }
  ];

  // --------------------------------------------------------------------------
  // Helper: Invalidate Prior Results (Image replacement & state changes)
  // --------------------------------------------------------------------------
  function invalidatePreviousResults() {
    currentSessionRequestId++;
    lastSessionData = null;
    if (btnExport) btnExport.style.display = 'none';
    if (resultsContent) {
      resultsContent.style.display = 'none';
      resultsContent.innerHTML = '';
    }
    if (resultsEmpty) resultsEmpty.style.display = 'flex';
  }

  // --------------------------------------------------------------------------
  // 1. Mode Switching (File Upload vs Live Laptop Camera)
  // --------------------------------------------------------------------------
  btnModeUpload.addEventListener('click', () => switchMode('upload'));
  btnModeWebcam.addEventListener('click', () => switchMode('webcam'));

  function switchMode(mode) {
    if (activeMode === mode) return;
    activeMode = mode;
    invalidatePreviousResults();

    btnModeUpload.classList.toggle('active', mode === 'upload');
    btnModeWebcam.classList.toggle('active', mode === 'webcam');

    if (mode === 'upload') {
      uploadSection.style.display = 'block';
      webcamSection.style.display = 'none';
      stopCameraStream();
    } else {
      uploadSection.style.display = 'none';
      webcamSection.style.display = 'block';
    }
    updateAnalyzeButtonState();
  }

  // --------------------------------------------------------------------------
  // 2. Module Selection
  // --------------------------------------------------------------------------
  moduleCards.forEach(card => {
    card.addEventListener('click', () => {
      const mod = card.getAttribute('data-module');
      selectModule(mod);
    });
  });

  function selectModule(mod) {
    currentModule = mod;
    invalidatePreviousResults();

    moduleCards.forEach(c => {
      const isSel = c.getAttribute('data-module') === mod;
      c.classList.toggle('selected', isSel);
      c.setAttribute('aria-pressed', isSel ? 'true' : 'false');
    });

    if (mod === 'anemia') {
      moduleBanner.className = 'alert-box alert-info';
      moduleBanner.innerHTML = `
        <strong>Active Module: Anemia Hemoglobin Regressor</strong><br>
        Permitted for experimental research use. You may upload clinical views or capture an experimental still frame via webcam.
      `;
      btnAnalyzeText.textContent = 'Run Research Inference';
      webcamDomainNotice.style.display = 'block';
      updateAnalyzeButtonState();
    } else if (mod === 'conjunctivitis') {
      moduleBanner.className = 'alert-box alert-danger';
      moduleBanner.innerHTML = `
        <strong>Reliability Gate Active: Conjunctivitis Classifier Blocked</strong><br>
        Layer 3.5 audit detected a 100% resolution/framing shortcut in the upstream dataset. All predictions are withheld to prevent non-biological diagnostic inferences.
      `;
      btnAnalyzeText.textContent = 'Inference Blocked (Validation Gate)';
      btnAnalyze.disabled = true;
      webcamDomainNotice.style.display = 'none';
      showBlockedOrUnavailableResults('conjunctivitis');
    } else if (mod === 'jaundice') {
      moduleBanner.className = 'alert-box alert-warning';
      moduleBanner.innerHTML = `
        <strong>Missing-Data Gate Active: Jaundice Classifier Unavailable</strong><br>
        No clinical dataset or laboratory serum bilirubin measurements exist in workspace. Synthetic predictions are prohibited.
      `;
      btnAnalyzeText.textContent = 'Module Unavailable (Missing Data)';
      btnAnalyze.disabled = true;
      webcamDomainNotice.style.display = 'none';
      showBlockedOrUnavailableResults('jaundice');
    }
  }

  // --------------------------------------------------------------------------
  // 3. Live Laptop Webcam Capture & Guidance (Layer 6)
  // --------------------------------------------------------------------------
  btnStartCamera.addEventListener('click', startCamera);
  btnCaptureStill.addEventListener('click', captureStillFrame);
  btnStopCamera.addEventListener('click', stopCameraStream);
  btnRecapture.addEventListener('click', recaptureFrame);
  webcamViewSelect.addEventListener('change', () => {
    invalidatePreviousResults();
    updateAnalyzeButtonState();
  });
  chkWebcamOverride.addEventListener('change', () => {
    invalidatePreviousResults();
    updateAnalyzeButtonState();
  });

  async function startCamera() {
    webcamStatusAlert.style.display = 'none';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraError('Unsupported Browser: Camera API (navigator.mediaDevices) is not supported in this environment.');
      return;
    }

    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamVideo.srcObject = mediaStream;
      await webcamVideo.play();

      btnStartCamera.style.display = 'none';
      btnStopCamera.style.display = 'inline-flex';
      btnCaptureStill.disabled = false;
    } catch (err) {
      console.error('Camera access error:', err);
      let errMsg = 'Failed to access camera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errMsg = 'Permission Denied: Camera access was blocked. Please grant camera permissions in your browser address bar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errMsg = 'No Camera Found: No camera device was detected on your laptop or system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errMsg = 'Camera in Use: Another application is already using the webcam.';
      }
      showCameraError(errMsg);
    }
  }

  function stopCameraStream() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => {
        try { track.stop(); } catch (e) {}
      });
      mediaStream = null;
    }
    if (webcamVideo) webcamVideo.srcObject = null;
    btnStartCamera.style.display = 'inline-flex';
    btnStopCamera.style.display = 'none';
    btnCaptureStill.disabled = true;
  }

  function showCameraError(msg) {
    webcamStatusAlert.style.display = 'block';
    webcamStatusAlert.className = 'alert-box alert-danger';
    webcamStatusAlert.innerHTML = `
      <strong>Camera Error:</strong> ${msg}
      <div style="margin-top: 6px; font-size: 11px;">You may switch to "File Upload Mode" above to test with existing files.</div>
    `;
  }

  async function captureStillFrame() {
    if (!webcamVideo.videoWidth || !webcamVideo.videoHeight) return;

    invalidatePreviousResults();

    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamCanvas.toBlob(async (blob) => {
      if (!blob) return;
      capturedWebcamBlob = blob;
      if (capturedWebcamUrl) URL.revokeObjectURL(capturedWebcamUrl);
      capturedWebcamUrl = URL.createObjectURL(blob);

      capturedStillThumb.src = capturedWebcamUrl;
      capturedStillCard.style.display = 'block';

      // Perform quality check on captured frame
      await runWebcamQualityCheck(blob);
      updateAnalyzeButtonState();
    }, 'image/jpeg', 0.95);
  }

  function recaptureFrame() {
    invalidatePreviousResults();
    if (capturedWebcamUrl) {
      URL.revokeObjectURL(capturedWebcamUrl);
      capturedWebcamUrl = null;
    }
    capturedWebcamBlob = null;
    capturedQuality = null;
    capturedStillCard.style.display = 'none';
    webcamGuidanceBox.innerHTML = '';
    updateAnalyzeButtonState();
  }

  async function runWebcamQualityCheck(blob) {
    webcamQualityMetrics.innerHTML = '<span>Analyzing capture quality...</span>';
    webcamGuidanceBox.innerHTML = '';

    const reqId = ++currentSessionRequestId;

    try {
      const formData = new FormData();
      formData.append('image', blob, 'webcam_capture.jpg');

      const res = await fetch('/api/quality-check', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error('Quality check failed');
      const data = await res.json();
      if (reqId !== currentSessionRequestId) return; // Stale check

      capturedQuality = data;

      const q = data.quality;
      const dims = data.dimensions || {};
      
      // Determine actual pixel dimensions with canvas fallback
      let width = null;
      let height = null;
      if (dims && typeof dims.width === 'number' && typeof dims.height === 'number') {
        width = dims.width;
        height = dims.height;
      } else if (webcamCanvas && webcamCanvas.width && webcamCanvas.height) {
        width = webcamCanvas.width;
        height = webcamCanvas.height;
      }
      const dimDisplay = (width && height) ? `${width}×${height} px` : 'Dimensions: Unavailable';

      let qualityPills = '';
      if (q) {
        const blurClass = q.is_blurry ? 'quality-pill warn' : 'quality-pill pass';
        const blurText = q.is_blurry ? 'Blurry' : 'Sharp';
        const expClass = (q.is_too_dark || q.is_too_bright) ? 'quality-pill warn' : 'quality-pill pass';
        const expText = q.is_too_dark ? 'Dark' : (q.is_too_bright ? 'Bright' : 'Exp OK');
        qualityPills = `
          <span class="${blurClass}">${blurText} (Laplacian: ${q.sharpness})</span>
          <span class="${expClass}">${expText} (L: ${q.brightness})</span>
        `;
      }

      webcamQualityMetrics.innerHTML = `
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 4px;">
          ${qualityPills}
          <span class="quality-pill pass">${dimDisplay}</span>
        </div>
      `;

      if (data.actionable_guidance && data.actionable_guidance.length > 0) {
        webcamGuidanceBox.innerHTML = `
          <strong>Actionable Capture Guidance:</strong>
          <ul>
            ${data.actionable_guidance.map(g => `<li>${g}</li>`).join('')}
          </ul>
          <div style="margin-top: 6px; font-size: 10px; color: var(--text-muted);">
            ${data.anatomical_disclaimer || ''}
          </div>
        `;
      }
    } catch (err) {
      if (reqId === currentSessionRequestId) {
        webcamQualityMetrics.innerHTML = '<span style="color: #f87171;">Quality check error</span>';
      }
    }
  }

  // --------------------------------------------------------------------------
  // 4. File Upload Drag & Drop Handlers
  // --------------------------------------------------------------------------
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFiles(Array.from(fileInput.files));
    }
  });

  async function handleFiles(newFiles) {
    const supportedExts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'];
    invalidatePreviousResults();

    for (const file of newFiles) {
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (!supportedExts.includes(ext)) {
        alert(`Unsupported file type: ${file.name}.`);
        continue;
      }
      let defaultView = 'unspecified';
      const fnLower = file.name.toLowerCase();
      if (fnLower.includes('palpebral') && fnLower.includes('forniceal')) defaultView = 'forniceal_palpebral';
      else if (fnLower.includes('palpebral')) defaultView = 'palpebral';
      else if (fnLower.includes('forniceal')) defaultView = 'forniceal';
      else if (fnLower.includes('eye') || fnLower.includes('raw')) defaultView = 'raw_eye';

      uploadedFiles.push({
        file,
        viewType: defaultView,
        previewUrl: URL.createObjectURL(file),
        quality: null,
        dimensions: null
      });
    }

    renderPreviews();
    await checkUploadedQuality();
    updateAnalyzeButtonState();
  }

  function renderPreviews() {
    if (uploadedFiles.length === 0) {
      previewsContainer.style.display = 'none';
      return;
    }
    previewsContainer.style.display = 'flex';
    previewsContainer.innerHTML = '';

    uploadedFiles.forEach((item, index) => {
      const row = document.createElement('div');
      row.className = 'preview-item';

      const thumb = document.createElement('img');
      thumb.className = 'preview-thumb';
      thumb.src = item.previewUrl;
      thumb.alt = item.file.name;

      const info = document.createElement('div');
      info.className = 'preview-info';

      const name = document.createElement('div');
      name.className = 'preview-name';
      name.textContent = item.file.name;

      const qSpan = document.createElement('div');
      qSpan.className = 'preview-quality';
      if (item.quality) {
        const q = item.quality;
        const blurClass = q.is_blurry ? 'quality-pill warn' : 'quality-pill pass';
        const expClass = (q.is_too_dark || q.is_too_bright) ? 'quality-pill warn' : 'quality-pill pass';
        const dimPill = item.dimensions ? `<span class="quality-pill pass">${item.dimensions.width}×${item.dimensions.height} px</span>` : '';
        qSpan.innerHTML = `
          <span class="${blurClass}">${q.is_blurry ? 'Blurry' : 'Sharp'} (${q.sharpness})</span>
          <span class="${expClass}">${q.is_too_dark ? 'Dark' : (q.is_too_bright ? 'Bright' : 'Exp OK')} (${q.brightness})</span>
          ${dimPill}
        `;
      } else {
        qSpan.innerHTML = `<span>Checking quality...</span>`;
      }
      info.appendChild(name);
      info.appendChild(qSpan);

      const selDiv = document.createElement('div');
      selDiv.className = 'view-selector';
      const sel = document.createElement('select');
      ANEMIA_VIEWS.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.label;
        if (opt.value === item.viewType) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener('change', (e) => {
        item.viewType = e.target.value;
        invalidatePreviousResults();
        updateAnalyzeButtonState();
      });
      selDiv.appendChild(sel);

      const btnDel = document.createElement('button');
      btnDel.className = 'btn-remove';
      btnDel.title = 'Remove image';
      btnDel.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      `;
      btnDel.addEventListener('click', () => {
        invalidatePreviousResults();
        URL.revokeObjectURL(item.previewUrl);
        uploadedFiles.splice(index, 1);
        renderPreviews();
        updateAnalyzeButtonState();
        evaluateQualityWarnings();
      });

      row.appendChild(thumb);
      row.appendChild(info);
      row.appendChild(selDiv);
      row.appendChild(btnDel);
      previewsContainer.appendChild(row);
    });
  }

  async function checkUploadedQuality() {
    const uninspected = uploadedFiles.filter(item => !item.quality);
    if (uninspected.length === 0) return;

    const reqId = currentSessionRequestId;

    for (const item of uninspected) {
      try {
        const formData = new FormData();
        formData.append('image', item.file);
        const res = await fetch('/api/quality-check', { method: 'POST', body: formData });
        if (res.ok) {
          const data = await res.json();
          if (reqId !== currentSessionRequestId) return;
          item.quality = data.quality;
          item.dimensions = data.dimensions;
        }
      } catch (err) {
        console.warn('Quality pre-check failed:', err);
      }
    }
    if (reqId === currentSessionRequestId) {
      renderPreviews();
      evaluateQualityWarnings();
    }
  }

  function evaluateQualityWarnings() {
    const warnings = [];
    uploadedFiles.forEach((item, idx) => {
      if (item.quality) {
        if (item.quality.is_blurry) warnings.push(`Image #${idx + 1} appears blurry (sharpness=${item.quality.sharpness}).`);
        if (item.quality.is_too_dark) warnings.push(`Image #${idx + 1} is underexposed.`);
        if (item.quality.is_too_bright) warnings.push(`Image #${idx + 1} is overexposed.`);
      }
    });

    if (warnings.length > 0) {
      qualityWarningBox.style.display = 'block';
      qualityWarningBox.innerHTML = `
        <strong>Capture Quality Warnings Detected:</strong>
        <ul style="margin-left: 18px; margin-top: 6px;">
          ${warnings.map(w => `<li>${w}</li>`).join('')}
        </ul>
      `;
    } else {
      qualityWarningBox.style.display = 'none';
      qualityWarningBox.innerHTML = '';
    }
  }

  function updateAnalyzeButtonState() {
    if (currentModule !== 'anemia') {
      btnAnalyze.disabled = true;
      return;
    }

    if (activeMode === 'upload') {
      btnAnalyze.disabled = (uploadedFiles.length === 0);
      btnAnalyzeText.textContent = 'Run Research Inference';
    } else {
      btnAnalyze.disabled = (capturedWebcamBlob === null);
      if (chkWebcamOverride.checked) {
        btnAnalyzeText.textContent = 'Run Experimental Webcam Test (Override)';
      } else {
        btnAnalyzeText.textContent = 'Run Webcam Validation Check (Gated)';
      }
    }
  }

  // --------------------------------------------------------------------------
  // 5. Analyze Action Execution
  // --------------------------------------------------------------------------
  btnAnalyze.addEventListener('click', async () => {
    if (currentModule !== 'anemia') return;
    setLoading(true);

    const thisRequestId = ++currentSessionRequestId;

    try {
      const formData = new FormData();
      formData.append('module', 'anemia');
      formData.append('capture_source', activeMode);

      if (activeMode === 'upload') {
        if (uploadedFiles.length === 0) return;
        formData.append('experimental_override', 'false');
        uploadedFiles.forEach(item => {
          formData.append('images', item.file);
          formData.append('view_types', item.viewType);
        });
      } else {
        // Webcam Mode
        if (!capturedWebcamBlob) return;
        formData.append('images', capturedWebcamBlob, 'webcam_capture.jpg');
        formData.append('view_types', webcamViewSelect.value);
        formData.append('experimental_override', chkWebcamOverride.checked ? 'true' : 'false');
      }

      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (thisRequestId !== currentSessionRequestId) {
        return; // Stale request, discard
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const result = await res.json();
      lastSessionData = result;

      if (result.operational_status === 'blocked_domain_validation') {
        renderWebcamBlockedDomainResults(result);
      } else {
        renderAnemiaResults(result);
      }
    } catch (err) {
      if (thisRequestId === currentSessionRequestId) {
        renderAnalysisError(err.message);
      }
    } finally {
      if (thisRequestId === currentSessionRequestId) {
        setLoading(false);
      }
    }
  });

  function setLoading(isLoading) {
    btnAnalyze.disabled = isLoading;
    btnSpinner.style.display = isLoading ? 'block' : 'none';
  }

  function renderAnalysisError(errorMsg) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    if (btnExport) btnExport.style.display = 'none';

    resultsContent.innerHTML = `
      <div class="result-card-main" style="border-color: rgba(239, 68, 68, 0.4);">
        <div class="result-header">
          <span style="font-size: 13px; font-weight: 600; color: #f87171;">ANALYSIS ERROR</span>
          <span class="status-badge blocked">FAILED</span>
        </div>
        <div style="margin: 14px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
          ${escapeHtml(errorMsg || 'An error occurred during inference.')}
        </div>
        <div class="alert-box alert-warning" style="font-size: 11px;">
          Please verify input image format, framing, and anatomical view type, then retry.
        </div>
      </div>
    `;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // --------------------------------------------------------------------------
  // 6. Results Rendering
  // --------------------------------------------------------------------------
  function renderWebcamBlockedDomainResults(session) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';

    resultsContent.innerHTML = `
      <div class="result-card-main" style="border-color: rgba(239, 68, 68, 0.4);">
        <div class="result-header">
          <span style="font-size: 13px; font-weight: 600; color: #f87171;">WEBCAM INFERENCE BLOCKED</span>
          <span class="status-badge blocked">BLOCKED (DOMAIN GATE)</span>
        </div>

        <div style="margin: 14px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
          <strong>Camera-Domain Validation Gate Enforced:</strong><br>
          The ResNet-50 Anemia model was trained exclusively on calibrated anterior-segment clinical research photography. Ordinary laptop webcams exhibit uncalibrated auto-gain, aggressive white-balance shifts, and compression artifacts that distort conjunctival erythema/pallor signals.
        </div>

        <div class="alert-box alert-danger">
          <h4>Clinical Safety Standard</h4>
          <p>
            Webcam predictions are withheld by default to prevent presenting unvalidated sensor readings as health indicators. A prospective clinical evaluation protocol is defined in <code>docs/WEBCAM_VALIDATION.md</code>.
          </p>
          <p style="margin-top: 6px;">
            To inspect raw experimental output for laboratory evaluation only, enable the <strong>Researcher Experimental Override</strong> checkbox in the camera panel.
          </p>
        </div>
      </div>
    `;
  }

  function renderAnemiaResults(session) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';

    const res = session;
    const viewCount = res.view_count || (res.model_output && res.model_output.length) || 1;
    const meanVal = res.view_mean !== null && res.view_mean !== undefined
      ? res.view_mean
      : (res.model_output && res.model_output[0] ? res.model_output[0].hb_pred_g_dl : 'N/A');
    const stdVal = res.view_std !== null && res.view_std !== undefined ? res.view_std : 0.0;

    const isSingleView = viewCount <= 1;
    const stdDisplay = (viewCount > 1 && res.view_std !== null && res.view_std !== undefined)
      ? `&plusmn;${stdVal} g/dL`
      : 'Not available — one view';
    const stdNote = (viewCount > 1)
      ? '*Reproducibility indicator across crops, NOT a confidence interval.'
      : '*Single view analyzed; cross-crop variation requires multiple captures.';

    const webcamOverrideBanner = (res.capture_source === 'webcam') ? `
      <div class="alert-box alert-danger" style="margin-bottom: 14px;">
        <strong>⚠️ EXPERIMENTAL WEBCAM OVERRIDE ACTIVE — UNVALIDATED OUTPUT:</strong><br>
        This estimate was derived from an uncalibrated laptop webcam. It is unvalidated and completely unsuitable for medical decisions.
      </div>
    ` : '';

    let viewsHtml = '';
    if (res.model_output && Array.isArray(res.model_output)) {
      viewsHtml = `
        <div style="margin-top: 14px;">
          <h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px;">Individual View Estimates:</h4>
          <div class="views-breakdown">
            ${res.model_output.map((p, i) => `
              <div class="view-row">
                <span>View #${p.view_index + 1} (<code>${p.view_type}</code>)</span>
                <strong>${p.hb_pred_g_dl} g/dL</strong>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    const warningsHtml = (res.reliability_warnings && res.reliability_warnings.length > 0) ? `
      <div class="alert-box alert-warning">
        <h4>Reliability &amp; Clinical Warnings</h4>
        <ul style="margin-left: 18px;">
          ${res.reliability_warnings.map(w => `<li>${w}</li>`).join('')}
        </ul>
      </div>
    ` : '';

    const limitationsHtml = (res.documented_limitations && res.documented_limitations.length > 0) ? `
      <div class="alert-box alert-info">
        <h4>Documented Model Limitations</h4>
        <ul style="margin-left: 18px;">
          ${res.documented_limitations.map(l => `<li>${l}</li>`).join('')}
        </ul>
      </div>
    ` : '';

    resultsContent.innerHTML = `
      ${webcamOverrideBanner}
      <div class="result-card-main">
        <div class="result-header">
          <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary);">
            ESTIMATED HEMOGLOBIN (EXPERIMENTAL &bull; ${res.capture_source ? res.capture_source.toUpperCase() : 'UPLOAD'})
          </span>
          <span class="status-badge available">RESEARCH ONLY</span>
        </div>

        <div class="result-value-display">
          <span class="result-number">${meanVal}</span>
          <span class="result-unit">g/dL</span>
        </div>

        <div class="result-stats-row">
          <div class="stat-item">Views Analyzed: <strong>${viewCount}</strong></div>
          <div class="stat-item">
            View-to-View Std: <strong>${stdDisplay}</strong>
            <span style="display: block; font-size: 10px; color: var(--text-muted); margin-top: 2px;">
              ${stdNote}
            </span>
          </div>
        </div>

        ${viewsHtml}
      </div>

      ${warningsHtml}
      ${limitationsHtml}

      <div class="alert-box" style="background: rgba(100, 116, 139, 0.1); border: 1px solid var(--border-subtle); color: var(--text-muted); font-size: 11px;">
        <strong>RESEARCH USE ONLY:</strong> This output is an experimental laboratory estimate. It does NOT replace standard venous blood hemocounter testing and must NOT be used for clinical triage, diagnosis, or treatment decisions.
      </div>
    `;
  }

  function showBlockedOrUnavailableResults(mod) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';

    if (mod === 'conjunctivitis') {
      lastSessionData = {
        session_id: `conjunctivitis_${Date.now()}`,
        timestamp: new Date().toISOString(),
        selected_module: 'conjunctivitis',
        operational_status: 'blocked_validation',
        inference_permitted: false,
        model_architecture: 'MobileNetV3-Small',
        model_output: null,
        target_unit: null,
        input_files: [],
        view_count: 0,
        view_mean: null,
        view_std: null,
        checkpoint_id: 'best_model_v2_clean.pth',
        capture_source: activeMode,
        reliability_warnings: [
          "BLOCKED: Model validation audit detected a 100% resolution/framing shortcut in the upstream dataset.",
          "Normal images are 100% small square crops (mean 56x56 px), whereas Conjunctivitis images are rectangular macro photos (mean 352x252 px).",
          "A metadata classifier using ONLY image dimensions achieves 100% test accuracy without inspecting any image pixels.",
          "The convolutional model cannot be trusted as a biological hyperemia detector; model output is withheld."
        ],
        documented_limitations: [
          "Model artifact is preserved offline for research evaluation (saved_models/conjunctivitis/best_model_v2_clean.pth).",
          "Revalidation requires acquiring matched-resolution anterior eye images with ophthalmologist grading."
        ],
        disclaimer: "RESEARCH USE ONLY: Experimental system. Not for medical diagnosis or clinical use."
      };

      resultsContent.innerHTML = `
        <div class="result-card-main" style="border-color: rgba(239, 68, 68, 0.3);">
          <div class="result-header">
            <span style="font-size: 13px; font-weight: 600; color: #f87171;">MODULE GATED — INFERENCE WITHHELD</span>
            <span class="status-badge blocked">BLOCKED (VALIDATION)</span>
          </div>
          <div style="margin: 16px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
            The Conjunctivitis convolutional network is locked. Our empirical Layer 3.5 audit demonstrated that class labels are 100% predictable using image dimensions alone (Rule: <code>width > 100</code> achieves 100% accuracy without pixels).
          </div>
          <div class="alert-box alert-danger">
            <h4>Data Shortcut Rationale</h4>
            <p>
              In the raw dataset, normal eye images are 100% square 56x56 px crops, whereas conjunctivitis images are rectangular macro photography. To prevent false biological diagnostic claims, predictions are strictly withheld.
            </p>
          </div>
        </div>
      `;
    } else if (mod === 'jaundice') {
      lastSessionData = {
        session_id: `jaundice_${Date.now()}`,
        timestamp: new Date().toISOString(),
        selected_module: 'jaundice',
        operational_status: 'unavailable_missing_data',
        inference_permitted: false,
        model_architecture: 'MobileNetV3-Small (Scaffold)',
        model_output: null,
        target_unit: null,
        input_files: [],
        view_count: 0,
        view_mean: null,
        view_std: null,
        checkpoint_id: null,
        capture_source: activeMode,
        reliability_warnings: [
          "UNAVAILABLE: No jaundice or scleral icterus dataset exists in the workspace.",
          "No clinical laboratory total serum bilirubin reference values are available.",
          "Clinical safety rules strictly forbid running placeholder inference or fabricating bilirubin values."
        ],
        documented_limitations: [
          "Model architecture (JaundiceClassifier) and preprocessing scaffolds are preserved in src/models/jaundice_model.py.",
          "Clinical data ingestion is required before training or inference can be activated."
        ],
        disclaimer: "RESEARCH USE ONLY: Experimental system. Not for medical diagnosis or clinical use."
      };

      resultsContent.innerHTML = `
        <div class="result-card-main" style="border-color: rgba(100, 116, 139, 0.3);">
          <div class="result-header">
            <span style="font-size: 13px; font-weight: 600; color: var(--text-muted);">MODULE UNAVAILABLE</span>
            <span class="status-badge unavailable">MISSING DATA</span>
          </div>
          <div style="margin: 16px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
            No raw ocular images with paired laboratory total serum bilirubin measurements exist in the local workspace. Synthetic predictions are prohibited.
          </div>
          <div class="alert-box alert-warning">
            <h4>Prerequisite Clinical Data</h4>
            <p>
              Activation requires an anterior eye dataset with paired laboratory total serum bilirubin reference measurements.
            </p>
          </div>
        </div>
      `;
    }
  }

  // --------------------------------------------------------------------------
  // 7. Reset Session Action (Fresh Session Handling)
  // --------------------------------------------------------------------------
  if (btnResetSession) {
    btnResetSession.addEventListener('click', resetSession);
  }

  function resetSession() {
    invalidatePreviousResults();
    stopCameraStream();

    // Revoke object URLs
    uploadedFiles.forEach(item => {
      if (item.previewUrl) {
        try { URL.revokeObjectURL(item.previewUrl); } catch (e) {}
      }
    });
    uploadedFiles = [];

    if (capturedWebcamUrl) {
      try { URL.revokeObjectURL(capturedWebcamUrl); } catch (e) {}
      capturedWebcamUrl = null;
    }
    capturedWebcamBlob = null;
    capturedQuality = null;

    if (fileInput) fileInput.value = '';
    if (previewsContainer) {
      previewsContainer.style.display = 'none';
      previewsContainer.innerHTML = '';
    }
    if (qualityWarningBox) {
      qualityWarningBox.style.display = 'none';
      qualityWarningBox.innerHTML = '';
    }

    if (capturedStillCard) capturedStillCard.style.display = 'none';
    if (webcamGuidanceBox) webcamGuidanceBox.innerHTML = '';
    if (webcamQualityMetrics) webcamQualityMetrics.innerHTML = '';
    if (webcamStatusAlert) webcamStatusAlert.style.display = 'none';
    if (chkWebcamOverride) chkWebcamOverride.checked = false;

    updateAnalyzeButtonState();
  }

  // --------------------------------------------------------------------------
  // 8. Session Export Action
  // --------------------------------------------------------------------------
  btnExport.addEventListener('click', async () => {
    if (!lastSessionData) return;

    try {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(lastSessionData)
      });

      if (!res.ok) throw new Error(`Export failed with HTTP ${res.status}`);

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `eyehealthai_${lastSessionData.selected_module || 'session'}_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      const blob = new Blob([JSON.stringify(lastSessionData, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `eyehealthai_session_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }
  });
});
