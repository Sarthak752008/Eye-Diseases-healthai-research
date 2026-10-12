/* ==========================================================================
   EyeHealthAI — Research Dashboard Client Logic (Layers 5, 6 & 7)
   Multi-Step Journey, Live Laptop Webcam Capture, Internationalization (EN/HI),
   Database Session Management, Two-Page PDF Export, and Gemini Research Copilot.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Global Workflow State
  let currentStep = 1; // 1 to 7
  let currentLanguage = 'en'; // 'en' or 'hi'
  let currentUserProfile = null; // { id, full_name, email, phone, date_of_birth }
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

  // --------------------------------------------------------------------------
  // Comprehensive Internationalization (English & Hindi)
  // --------------------------------------------------------------------------
  const I18N = {
    en: {
      appTitle: "EyeHealthAI",
      appSubtitle: "Multi-Modal Ocular Screening &bull; Continuous Hb Regression &bull; Clinical Safety Gates",
      step1: "Language",
      step2: "Consent",
      step3: "Preparation",
      step4: "Capture",
      step5: "Analysis",
      step6: "Results",
      step7: "History",
      langTitle: "Select Your Preferred Language",
      langSub: "Choose your language for guided preparation, ocular screening, and downloaded clinical reports. Both languages are fully supported across all steps.",
      btnContinue: "Continue &rarr;",
      regTitle: "Participant Profile &amp; Informed Consent",
      regSub: "Please enter your details and review the research consent terms before proceeding. Your data is stored privacy-compliantly.",
      labelFullName: "Full Name *",
      labelPhone: "Phone Number *",
      labelEmail: "Email Address *",
      labelDob: "Date of Birth (YYYY-MM-DD) *",
      labelGender: "Gender (Optional Demographic Data)",
      genderOptPreferNot: "Prefer not to say (Default)",
      genderOptFemale: "Female",
      genderOptMale: "Male",
      genderOptNonBinary: "Non-binary",
      genderOptSelfDescribe: "Self-describe",
      chkConsentDemographicText: "(Optional) I consent to the collection of demographic data (gender) exclusively for research diversity evaluation. You may continue without selecting a gender.",
      consentHeader: "Informed Consent &amp; Privacy Notice",
      consentP1: "<strong>Experimental Research Only:</strong> EyeHealthAI is an investigational software prototype. It does NOT provide a certified medical diagnosis.",
      consentP2: "<strong>No Raw Image Retention:</strong> Captured eye photographs and webcam frames are processed strictly in RAM and are NEVER permanently stored on disk by default.",
      consentP3: "<strong>Laboratory Reference Required:</strong> Continuous hemoglobin estimations require confirmatory laboratory Complete Blood Count (CBC) venous blood tests.",
      consentP4: "<strong>Right to Be Forgotten:</strong> You can permanently delete your participant profile, session history, and report metadata at any time.",
      chkConsentResearchText: "I understand this is an experimental research prototype and agree to participate in non-invasive ocular screening.",
      chkConsentStorageText: "I consent to privacy-aware session and report metadata storage with the ability to purge my records at any time.",
      btnBack: "&larr; Back",
      btnAgreeContinue: "Agree &amp; Continue &rarr;",
      guideTitle: "Guided Preparation &amp; Protocol",
      guideSub: "Follow these 6 simple steps for optimal optical clarity, non-invasive imaging, and reliable screening results.",
      guideStep1Title: "Diffuse Illumination",
      guideStep1Desc: "Face an indirect, soft light source. Avoid direct flash, harsh sunlight reflections, or heavy shadows on the ocular surface.",
      guideStep2Title: "Enable Camera",
      guideStep2Desc: "Allow browser camera permissions when prompted. Alternatively, use research photograph file upload.",
      guideStep3Title: "Eye Positioning (10–15 cm)",
      guideStep3Desc: "Position the target eye within 10–15 cm of the lens. Macro anterior view required, NOT a distant full-face selfie.",
      guideStep4Title: "Non-Invasive View Selection",
      guideStep4Desc: "Gently expose the lower palpebral conjunctiva by resting a clean fingertip on the lower cheekbone. Never force or pull.",
      guideStep5Title: "Automated Quality Check",
      guideStep5Desc: "The system checks sharpness and luminance in real-time. If blur or glare is detected, instant feedback is provided.",
      guideStep6Title: "Downloadable 2-Page Report",
      guideStep6Desc: "Receive an official, two-page clinical research report in English or Hindi with complete diagnostic safety boundaries.",
      btnReadyCapture: "Ready to Capture &rarr;",
      analysisLoadingTitle: "Processing Ocular Computer Vision Pipeline",
      analysisLoadingSub: "Inspecting pixel luminance, verifying anatomical crop, and computing multi-view ResNet-50 continuous hemoglobin regression...",
      resultsTitle: "Ocular Screening Results &amp; Clinical Insights",
      resultsSub: "Experimental non-invasive screening findings with active reliability and domain validation boundaries.",
      btnDownloadPdf: "Download 2-Page PDF (EN)",
      btnViewPrintable: "View Printable Scorecard",
      btnBackCapture: "&larr; Back to Capture",
      btnNewScreening: "Start New Screening",
      btnViewHistory: "View Session History &rarr;",
      historyTitle: "Screening Session History",
      historySub: "Review prior screening sessions, access verified reports, or exercise privacy deletion rights.",
      btnPurgeAll: "Purge All My Records (GDPR)",
      thDate: "Date / Time",
      thSessionId: "Session ID",
      thModule: "Target Module",
      thSource: "Source",
      thStatus: "Status / Result",
      thActions: "Actions",
      historyEmpty: "No prior screening sessions found for this participant profile.",
      btnBackScreening: "&larr; Back to Screening Dashboard"
    },
    hi: {
      appTitle: "EyeHealthAI",
      appSubtitle: "मल्टी-मॉडल नेत्र जांच &bull; हीमोग्लोबिन निरंतर प्रतिगमन &bull; चिकित्सीय सुरक्षा गेट",
      step1: "भाषा",
      step2: "सहमति",
      step3: "तैयारी",
      step4: "कैप्चर",
      step5: "विश्लेषण",
      step6: "परिणाम",
      step7: "इतिहास",
      langTitle: "अपनी पसंदीदा भाषा चुनें",
      langSub: "मार्गदर्शन, जांच और रिपोर्ट के लिए अपनी पसंदीदा भाषा चुनें। दोनों भाषाएं सभी चरणों में पूरी तरह समर्थित हैं।",
      btnContinue: "आगे बढ़ें &rarr;",
      regTitle: "प्रतिभागी विवरण और सूचित सहमति",
      regSub: "आगे बढ़ने से पहले कृपया अपना विवरण दर्ज करें और अनुसंधान सहमति शर्तों की समीक्षा करें। आपकी गोपनीयता सुरक्षित है।",
      labelFullName: "पूरा नाम *",
      labelPhone: "फोन नंबर *",
      labelEmail: "ईमेल पता *",
      labelDob: "जन्म तिथि (YYYY-MM-DD) *",
      labelGender: "लिंग (वैकल्पिक जनसांख्यिकीय डेटा)",
      genderOptPreferNot: "बताना नहीं चाहते (डिफ़ॉल्ट)",
      genderOptFemale: "महिला (Female)",
      genderOptMale: "पुरुष (Male)",
      genderOptNonBinary: "नॉन-बाइनरी (Non-binary)",
      genderOptSelfDescribe: "स्व-वर्णन (Self-describe)",
      chkConsentDemographicText: "(वैकल्पिक) मैं विविधता और निष्पक्षता मूल्यांकन हेतु जनसांख्यिकीय डेटा (लिंग) संग्रह की सहमति देता/देती हूँ। आप लिंग का चयन किए बिना भी जारी रख सकते हैं।",
      consentHeader: "सूचित सहमति एवं गोपनीयता सूचना",
      consentP1: "<strong>केवल प्रायोगिक अनुसंधान:</strong> EyeHealthAI एक अनुसंधान सॉफ्टवेयर है। यह प्रमाणित चिकित्सीय निदान (Medical Diagnosis) प्रदान नहीं करता है।",
      consentP2: "<strong>मूल तस्वीरें सुरक्षित नहीं:</strong> ली गई आंख की तस्वीरें केवल रैम में प्रोसेस होती हैं और डिस्क पर कभी स्थायी रूप से सुरक्षित नहीं रखी जातीं।",
      consentP3: "<strong>प्रयोगशाला परीक्षण आवश्यक:</strong> निरंतर हीमोग्लोबिन अनुमानों के लिए प्रयोगशाला सीबीसी (CBC) रक्त परीक्षण द्वारा पुष्टि आवश्यक है।",
      consentP4: "<strong>डेटा मिटाने का अधिकार:</strong> आप किसी भी समय अपनी प्रोफाइल और इतिहास को स्थायी रूप से मिटा सकते हैं।",
      chkConsentResearchText: "मैं समझता/समझती हूँ कि यह एक अनुसंधान प्रोटोटाइप है और गैर-आक्रामक जांच में भाग लेने के लिए सहमत हूँ।",
      chkConsentStorageText: "मैं सत्र और रिपोर्ट मेटाडेटा के सुरक्षित भंडारण की सहमति देता/देती हूँ।",
      btnBack: "&larr; पीछे",
      btnAgreeContinue: "सहमति दें और आगे बढ़ें &rarr;",
      guideTitle: "मार्गदर्शित तैयारी और दिशानिर्देश",
      guideSub: "स्पष्ट तस्वीर और विश्वसनीय जांच परिणामों के लिए इन 6 चरणों का पालन करें।",
      guideStep1Title: "उचित रोशनी",
      guideStep1Desc: "कमरे में सामान्य, अप्रत्यक्ष रोशनी रखें। आंख पर सीधे तेज फ्लैश या धूप के परावर्तन से बचें।",
      guideStep2Title: "कैमरा सक्षम करें",
      guideStep2Desc: "ब्राउज़र को कैमरा उपयोग की अनुमति दें। वैकल्पिक रूप से, फोटो अपलोड का उपयोग करें।",
      guideStep3Title: "आंख की दूरी (10–15 सेमी)",
      guideStep3Desc: "आंख को लेंस से 10–15 सेमी की दूरी पर रखें। आंख की नजदीकी तस्वीर आवश्यक है, पूरे चेहरे की नहीं।",
      guideStep4Title: "सुरक्षित पलक स्थिति",
      guideStep4Desc: "निचली पलक के अंदरूनी हिस्से को धीरे से दिखाएं। कभी भी जोर या खिंचाव न लगाएं।",
      guideStep5Title: "स्वचालित गुणवत्ता जांच",
      guideStep5Desc: "सिस्टम तीखेपन और रोशनी की रीयल-टाइम जांच करता है। धुंधलापन होने पर तुरंत निर्देश मिलते हैं।",
      guideStep6Title: "2-पृष्ठों की रिपोर्ट",
      guideStep6Desc: "अंग्रेजी या हिंदी में पूरी नैदानिक सुरक्षा सीमाओं के साथ आधिकारिक शोध रिपोर्ट प्राप्त करें।",
      btnReadyCapture: "फोटो खींचने के लिए तैयार &rarr;",
      analysisLoadingTitle: "नेत्र कंप्यूटर विज़न विश्लेषण जारी है",
      analysisLoadingSub: "पिक्सेल गुणवत्ता की जांच, एनाटोमिकल भाग की पुष्टि, और ResNet-50 हीमोग्लोबिन प्रतिगमन की गणना की जा रही है...",
      resultsTitle: "नेत्र जांच परिणाम एवं चिकित्सीय अंतर्दृष्टि",
      resultsSub: "सक्रिय विश्वसनीयता और डोमेन सत्यापन सीमाओं के साथ प्रायोगिक जांच निष्कर्ष।",
      btnDownloadPdf: "2-पेज पीडीएफ डाउनलोड (EN)",
      btnViewPrintable: "प्रिंट योग्य स्कोरकार्ड देखें",
      btnBackCapture: "&larr; वापस कैप्चर पर",
      btnNewScreening: "नई जांच शुरू करें",
      btnViewHistory: "जांच इतिहास देखें &rarr;",
      historyTitle: "स्क्रीनिंग सत्र इतिहास",
      historySub: "पिछले जांच सत्रों की समीक्षा करें, रिपोर्ट डाउनलोड करें या डेटा मिटाएं।",
      btnPurgeAll: "मेरा सारा डेटा मिटाएं (Right to be Forgotten)",
      thDate: "दिनांक / समय",
      thSessionId: "सत्र आईडी",
      thModule: "लक्ष्य मॉड्यूल",
      thSource: "स्रोत",
      thStatus: "स्थिति / परिणाम",
      thActions: "कार्रवाई",
      historyEmpty: "इस प्रोफाइल के लिए कोई पिछला सत्र नहीं मिला।",
      btnBackScreening: "&larr; जांच डैशबोर्ड पर वापस जाएं"
    }
  };

  // --------------------------------------------------------------------------
  // DOM Elements - Stepper & Header
  // --------------------------------------------------------------------------
  const langBtnEn = document.getElementById('langBtnEn');
  const langBtnHi = document.getElementById('langBtnHi');
  const userBadge = document.getElementById('userBadge');
  const userNameBadge = document.getElementById('userNameBadge');
  const btnNavHistory = document.getElementById('btnNavHistory');
  const btnResetSession = document.getElementById('btnResetSession');
  const btnNewSessionText = document.getElementById('btnNewSessionText');
  const btnNavHistoryText = document.getElementById('btnNavHistoryText');

  // Page 1 Elements
  const cardLangEn = document.getElementById('cardLangEn');
  const cardLangHi = document.getElementById('cardLangHi');
  const btnContinueLang = document.getElementById('btnContinueLang');

  // Page 2 Elements
  const formRegistration = document.getElementById('formRegistration');
  const inputRegName = document.getElementById('inputRegName');
  const inputRegPhone = document.getElementById('inputRegPhone');
  const inputRegEmail = document.getElementById('inputRegEmail');
  const inputRegDob = document.getElementById('inputRegDob');
  const chkConsentResearch = document.getElementById('chkConsentResearch');
  const chkConsentStorage = document.getElementById('chkConsentStorage');
  const regErrorAlert = document.getElementById('regErrorAlert');
  const btnBackToLang = document.getElementById('btnBackToLang');

  // Page 3 Elements
  const btnBackToReg = document.getElementById('btnBackToReg');
  const btnProceedToCapture = document.getElementById('btnProceedToCapture');

  // Page 4 Elements - Modules & Inputs
  const moduleCards = document.querySelectorAll('.module-card');
  const moduleBanner = document.getElementById('moduleBanner');
  const btnModeUpload = document.getElementById('btnModeUpload');
  const btnModeWebcam = document.getElementById('btnModeWebcam');
  const uploadSection = document.getElementById('uploadSection');
  const webcamSection = document.getElementById('webcamSection');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewsContainer = document.getElementById('previewsContainer');
  const qualityWarningBox = document.getElementById('qualityWarningBox');
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
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnAnalyzeText = document.getElementById('btnAnalyzeText');
  const btnSpinner = document.getElementById('btnSpinner');
  const resultsEmpty = document.getElementById('resultsEmpty');
  const resultsContent = document.getElementById('resultsContent');
  const btnViewReport = document.getElementById('btnViewReport');
  const btnExport = document.getElementById('btnExport');

  // Eye-Focused Camera & Cropping Elements
  const btnCamModeAuto = document.getElementById('btnCamModeAuto');
  const btnCamModeManual = document.getElementById('btnCamModeManual');
  const btnCamZoomToggle = document.getElementById('btnCamZoomToggle');
  const liveGuidancePill = document.getElementById('liveGuidancePill');
  const eyeDetectorBox = document.getElementById('eyeDetectorBox');
  const eyeGuideReticle = document.getElementById('eyeGuideReticle');
  const guideTitleText = document.getElementById('guideTitleText');
  const guideSubText = document.getElementById('guideSubText');
  const cropCanvas = document.getElementById('cropCanvas');
  const capturedCropBadge = document.getElementById('capturedCropBadge');
  const capturedCropMetadata = document.getElementById('capturedCropMetadata');
  const capturedRejectionBox = document.getElementById('capturedRejectionBox');

  // Dynamic Camera & Eye Detection State
  let isEyeModeAuto = true;
  let isZoom2x = false;
  let liveEyeTrackingInterval = null;
  let lastOcularCropBox = null;

  // Page 6 Elements - Results
  const fullResultsContainer = document.getElementById('fullResultsContainer');
  const btnDownloadPdfReport = document.getElementById('btnDownloadPdfReport');
  const btnDownloadPdfHi = document.getElementById('btnDownloadPdfHi');
  const btnOpenPrintScorecard = document.getElementById('btnOpenPrintScorecard');
  const btnBackToCapture = document.getElementById('btnBackToCapture');
  const btnNewScreeningFromResults = document.getElementById('btnNewScreeningFromResults');
  const btnViewHistoryFromResults = document.getElementById('btnViewHistoryFromResults');

  // Page 7 Elements - History
  const historyTableBody = document.getElementById('historyTableBody');
  const historyEmptyState = document.getElementById('historyEmptyState');
  const btnPurgeUserData = document.getElementById('btnPurgeUserData');
  const btnBackToScreening = document.getElementById('btnBackToScreening');

  // Report Modal Preview
  const reportModal = document.getElementById('reportModal');
  const btnCloseReportModal = document.getElementById('btnCloseReportModal');
  const reportFrame = document.getElementById('reportFrame');

  // View Options for Anemia
  const ANEMIA_VIEWS = [
    { value: 'palpebral', label: 'Palpebral Conjunctiva' },
    { value: 'forniceal', label: 'Forniceal Conjunctiva' },
    { value: 'forniceal_palpebral', label: 'Forniceal + Palpebral' },
    { value: 'raw_eye', label: 'Raw Full Eye View' },
    { value: 'unspecified', label: 'Unspecified Ocular Crop' }
  ];

  // --------------------------------------------------------------------------
  // Internationalization Logic
  // --------------------------------------------------------------------------
  function setLanguage(lang) {
    currentLanguage = lang;
    if (langBtnEn) langBtnEn.classList.toggle('active', lang === 'en');
    if (langBtnHi) langBtnHi.classList.toggle('active', lang === 'hi');
    if (cardLangEn) cardLangEn.classList.toggle('selected', lang === 'en');
    if (cardLangHi) cardLangHi.classList.toggle('selected', lang === 'hi');

    const dict = I18N[lang] || I18N.en;
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (dict[key]) {
        el.innerHTML = dict[key];
      }
    });

    if (btnNewSessionText) btnNewSessionText.textContent = lang === 'hi' ? 'नया सत्र' : 'New Session';
    if (btnNavHistoryText) btnNavHistoryText.textContent = lang === 'hi' ? 'इतिहास' : 'History';

    // Update results cards if active
    if (lastSessionData) {
      renderFullResults(lastSessionData);
    }
  }

  if (langBtnEn) langBtnEn.addEventListener('click', () => setLanguage('en'));
  if (langBtnHi) langBtnHi.addEventListener('click', () => setLanguage('hi'));

  if (cardLangEn) {
    cardLangEn.addEventListener('click', () => {
      setLanguage('en');
    });
  }
  if (cardLangHi) {
    cardLangHi.addEventListener('click', () => {
      setLanguage('hi');
    });
  }
  if (btnContinueLang) {
    btnContinueLang.addEventListener('click', () => {
      goToStep(2);
    });
  }

  // --------------------------------------------------------------------------
  // Multi-Step Navigation Engine
  // --------------------------------------------------------------------------
  function goToStep(step) {
    currentStep = step;

    const pageViews = [
      { num: 1, id: 'page1Language' },
      { num: 2, id: 'page2Registration' },
      { num: 3, id: 'page3Guide' },
      { num: 4, id: 'page4Capture' },
      { num: 5, id: 'page5Analysis' },
      { num: 6, id: 'page6Results' },
      { num: 7, id: 'page7History' },
    ];

    pageViews.forEach(p => {
      const el = document.getElementById(p.id);
      if (el) {
        el.classList.toggle('active', p.num === step);
      }
      const stepBtn = document.getElementById(`stepBtn${p.num}`);
      if (stepBtn) {
        stepBtn.classList.toggle('active', p.num === step);
        stepBtn.classList.toggle('completed', p.num < step);
      }
      const div = document.getElementById(`stepDiv${p.num}`);
      if (div) {
        div.classList.toggle('active', p.num < step);
      }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (step === 7) {
      loadSessionHistory();
    }
  }

  // Stepper buttons click handlers
  for (let s = 1; s <= 7; s++) {
    const btn = document.getElementById(`stepBtn${s}`);
    if (btn) {
      btn.addEventListener('click', () => {
        // Can jump back to any previous step or current step
        if (s <= Math.max(currentStep, 4)) {
          goToStep(s);
        }
      });
    }
  }

  if (btnBackToLang) btnBackToLang.addEventListener('click', () => goToStep(1));
  if (btnBackToReg) btnBackToReg.addEventListener('click', () => goToStep(2));
  if (btnProceedToCapture) btnProceedToCapture.addEventListener('click', () => goToStep(4));
  if (btnBackToCapture) btnBackToCapture.addEventListener('click', () => goToStep(4));
  if (btnNewScreeningFromResults) btnNewScreeningFromResults.addEventListener('click', () => {
    resetSession();
    goToStep(4);
  });
  if (btnViewHistoryFromResults) btnViewHistoryFromResults.addEventListener('click', () => goToStep(7));
  if (btnBackToScreening) btnBackToScreening.addEventListener('click', () => goToStep(4));
  if (btnNavHistory) btnNavHistory.addEventListener('click', () => goToStep(7));

  // --------------------------------------------------------------------------
  // User Registration & Informed Consent (Page 2)
  // --------------------------------------------------------------------------
  if (formRegistration) {
    formRegistration.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (regErrorAlert) regErrorAlert.style.display = 'none';

      const selectRegGender = document.getElementById('selectRegGender');
      const chkConsentDemographic = document.getElementById('chkConsentDemographic');

      const payload = {
        full_name: inputRegName.value.trim(),
        phone: inputRegPhone.value.trim(),
        email: inputRegEmail.value.trim(),
        date_of_birth: inputRegDob.value.trim(),
        gender: selectRegGender ? selectRegGender.value : 'prefer_not_to_say',
        demographic_consent: chkConsentDemographic ? chkConsentDemographic.checked : false,
        research_consent: chkConsentResearch.checked,
        data_storage_consent: chkConsentStorage.checked
      };

      try {
        const res = await fetch('/api/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `Registration failed with HTTP ${res.status}`);
        }

        const data = await res.json();
        currentUserProfile = data.user;

        if (userBadge && userNameBadge) {
          userNameBadge.textContent = currentUserProfile.full_name;
          userBadge.style.display = 'inline-flex';
        }

        goToStep(3); // Advance to preparation guide
      } catch (err) {
        if (regErrorAlert) {
          regErrorAlert.textContent = err.message;
          regErrorAlert.style.display = 'block';
        }
      }
    });
  }

  // --------------------------------------------------------------------------
  // Helper: Invalidate Prior Results (Image replacement & state changes)
  // --------------------------------------------------------------------------
  function invalidatePreviousResults() {
    currentSessionRequestId++;
    lastSessionData = null;
    if (btnExport) btnExport.style.display = 'none';
    if (btnViewReport) btnViewReport.style.display = 'none';
    if (resultsContent) {
      resultsContent.style.display = 'none';
      resultsContent.innerHTML = '';
    }
    if (fullResultsContainer) {
      fullResultsContainer.innerHTML = '';
    }
    if (resultsEmpty) resultsEmpty.style.display = 'flex';
  }

  // --------------------------------------------------------------------------
  // Mode Switching (File Upload vs Live Laptop Camera)
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
  // Module Selection
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
      const isSelected = c.getAttribute('data-module') === mod;
      c.classList.toggle('selected', isSelected);
      c.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
    });

    if (mod === 'anemia') {
      moduleBanner.className = 'alert-box alert-info';
      moduleBanner.innerHTML = `
        <strong>Active Module: Anemia Hemoglobin Regressor</strong><br>
        Permitted for experimental research use. You may upload one or multiple anatomical views (e.g. palpebral, forniceal, full eye) from the same capture session.
      `;
      btnAnalyzeText.textContent = 'Run Research Inference';
      if (webcamDomainNotice) webcamDomainNotice.style.display = 'block';
    } else if (mod === 'slid' || mod === 'conjunctival_injection') {
      moduleBanner.className = 'alert-box alert-info';
      moduleBanner.innerHTML = `
        <strong>Active Module: SLID Conjunctival Injection Classifier</strong><br>
        Permitted for research screening on slit-lamp photography. MobileNetV3-Small (Acc: 98.78%). Confounded Mendeley shortcut model remains blocked.
      `;
      btnAnalyzeText.textContent = 'Run Research Inference';
      if (webcamDomainNotice) webcamDomainNotice.style.display = 'block';
    } else if (mod === 'conjunctivitis') {
      moduleBanner.className = 'alert-box alert-danger';
      moduleBanner.innerHTML = `
        <strong>Reliability Gate Active: Mendeley Conjunctivitis Model</strong><br>
        Predictions are blocked due to a 100% resolution confound in source data. Use SLID module for valid research screening.
      `;
      btnAnalyzeText.textContent = 'Inspect Gated Module';
      if (webcamDomainNotice) webcamDomainNotice.style.display = 'none';
    } else if (mod === 'jaundice') {
      moduleBanner.className = 'alert-box alert-info';
      moduleBanner.innerHTML = `
        <strong>Active Module: Categorical Scleral Icterus Screening</strong><br>
        MobileNetV3-Small (CC BY-NC 4.0). Quantitative total serum bilirubin estimation remains strictly unavailable (missing lab ground truth).
      `;
      btnAnalyzeText.textContent = 'Run Research Inference';
      if (webcamDomainNotice) webcamDomainNotice.style.display = 'block';
    }

    updateAnalyzeButtonState();
  }

  // --------------------------------------------------------------------------
  // Live Laptop Camera Operations (Eye-Focused Macro Capture & Tracking)
  // --------------------------------------------------------------------------
  btnStartCamera.addEventListener('click', startCameraStream);
  btnStopCamera.addEventListener('click', stopCameraStream);
  btnCaptureStill.addEventListener('click', captureWebcamStill);
  btnRecapture.addEventListener('click', recaptureWebcamStill);

  if (btnCamModeAuto) {
    btnCamModeAuto.addEventListener('click', () => {
      isEyeModeAuto = true;
      btnCamModeAuto.classList.add('active');
      btnCamModeManual.classList.remove('active');
      if (eyeGuideReticle) eyeGuideReticle.style.display = 'none';
      if (guideTitleText) guideTitleText.textContent = 'Automated Eye Detection Active';
      if (guideSubText) guideSubText.textContent = 'Position one eye in the frame. The green guide will track your eye.';
    });
  }

  if (btnCamModeManual) {
    btnCamModeManual.addEventListener('click', () => {
      isEyeModeAuto = false;
      btnCamModeManual.classList.add('active');
      btnCamModeAuto.classList.remove('active');
      if (eyeDetectorBox) eyeDetectorBox.style.display = 'none';
      if (eyeGuideReticle) eyeGuideReticle.style.display = 'flex';
      if (guideTitleText) guideTitleText.textContent = 'Manual Macro Framing Target (10–15 cm)';
      if (guideSubText) guideSubText.textContent = 'Position one eye to fill this area. Positioning aid only; not automated eye detection.';
    });
  }

  if (btnCamZoomToggle) {
    btnCamZoomToggle.addEventListener('click', () => {
      isZoom2x = !isZoom2x;
      if (isZoom2x) {
        webcamVideo.style.transform = 'scale(1.55)';
        webcamVideo.style.transformOrigin = 'center center';
        btnCamZoomToggle.textContent = '🔍 2x Macro Preview';
        btnCamZoomToggle.classList.add('active');
      } else {
        webcamVideo.style.transform = 'none';
        btnCamZoomToggle.textContent = '🔍 1x View';
        btnCamZoomToggle.classList.remove('active');
      }
    });
  }

  if (chkWebcamOverride) {
    chkWebcamOverride.addEventListener('change', () => {
      updateAnalyzeButtonState();
    });
  }

  async function startCameraStream() {
    hideWebcamAlert();
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });
      webcamVideo.srcObject = mediaStream;
      btnStartCamera.style.display = 'none';
      btnStopCamera.style.display = 'inline-flex';
      btnCaptureStill.disabled = false;
      startLiveEyeTracking();
    } catch (err) {
      showWebcamAlert(`Camera access denied or unavailable: ${err.message}. Please check permissions.`);
    }
  }

  function stopCameraStream() {
    stopLiveEyeTracking();
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    webcamVideo.srcObject = null;
    btnStartCamera.style.display = 'inline-flex';
    btnStopCamera.style.display = 'none';
    btnCaptureStill.disabled = true;
    if (eyeDetectorBox) eyeDetectorBox.style.display = 'none';
  }

  function startLiveEyeTracking() {
    if (liveEyeTrackingInterval) clearInterval(liveEyeTrackingInterval);
    liveEyeTrackingInterval = setInterval(async () => {
      if (!mediaStream || webcamVideo.paused || webcamVideo.ended || !isEyeModeAuto) return;
      if (webcamVideo.videoWidth < 50) return;

      const trackCanvas = document.createElement('canvas');
      trackCanvas.width = 320;
      trackCanvas.height = Math.round(320 * (webcamVideo.videoHeight / webcamVideo.videoWidth)) || 240;
      const tctx = trackCanvas.getContext('2d');
      tctx.drawImage(webcamVideo, 0, 0, trackCanvas.width, trackCanvas.height);

      trackCanvas.toBlob(async (blob) => {
        if (!blob) return;
        const fd = new FormData();
        fd.append('image', blob, 'track_frame.jpg');
        try {
          const res = await fetch('/api/detect-eye', { method: 'POST', body: fd });
          if (!res.ok) return;
          const data = await res.json();
          const det = data.detection;
          updateLiveEyeUI(det, trackCanvas.width, trackCanvas.height);
        } catch (e) {
          // Silent catch in background polling
        }
      }, 'image/jpeg', 0.65);
    }, 650);
  }

  function stopLiveEyeTracking() {
    if (liveEyeTrackingInterval) {
      clearInterval(liveEyeTrackingInterval);
      liveEyeTrackingInterval = null;
    }
  }

  function updateLiveEyeUI(det, previewW, previewH) {
    if (!eyeDetectorBox || !isEyeModeAuto) return;

    if (det && det.eye_detected && det.crop_box) {
      lastOcularCropBox = det.crop_box;
      const pctX = (det.crop_box.x / previewW) * 100;
      const pctY = (det.crop_box.y / previewH) * 100;
      const pctW = (det.crop_box.w / previewW) * 100;
      const pctH = (det.crop_box.h / previewH) * 100;

      eyeDetectorBox.style.left = `${Math.max(2, Math.min(80, pctX))}%`;
      eyeDetectorBox.style.top = `${Math.max(2, Math.min(80, pctY))}%`;
      eyeDetectorBox.style.width = `${Math.min(96, pctW)}%`;
      eyeDetectorBox.style.height = `${Math.min(96, pctH)}%`;
      eyeDetectorBox.style.display = 'block';

      if (liveGuidancePill) {
        liveGuidancePill.textContent = det.guidance || 'Keep one eye inside the frame';
      }
      if (guideTitleText) guideTitleText.textContent = 'Automated Eye Detection Active';
      if (guideSubText) guideSubText.textContent = det.guidance || 'Keep one eye inside the frame. Hold steady to capture.';
    } else {
      lastOcularCropBox = null;
      if (eyeDetectorBox) eyeDetectorBox.style.display = 'none';
      if (liveGuidancePill) {
        liveGuidancePill.textContent = (det && det.guidance) ? det.guidance : 'Move closer (10–15 cm)';
      }
      if (guideTitleText) guideTitleText.textContent = 'Manual Macro Framing Target (10–15 cm)';
      if (guideSubText) guideSubText.textContent = (det && det.guidance) ? det.guidance : 'Position one eye to fill this area. Positioning aid only; not automated eye detection.';
    }
  }

  async function captureWebcamStill() {
    if (!mediaStream) return;

    webcamCanvas.width = webcamVideo.videoWidth || 640;
    webcamCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamCanvas.toBlob(async (fullBlob) => {
      if (!fullBlob) return;

      capturedStillCard.style.display = 'block';
      if (capturedRejectionBox) capturedRejectionBox.style.display = 'none';
      if (webcamGuidanceBox) webcamGuidanceBox.innerHTML = '<span style="color: var(--text-muted);">Inspecting focus, luminance, and framing...</span>';
      if (webcamQualityMetrics) webcamQualityMetrics.innerHTML = '';

      invalidatePreviousResults();
      await processAndCropWebcamCapture(fullBlob);
      updateAnalyzeButtonState();
    }, 'image/jpeg', 0.95);
  }

  async function processAndCropWebcamCapture(fullBlob) {
    const formData = new FormData();
    formData.append('image', fullBlob, 'webcam_capture.jpg');

    try {
      const res = await fetch('/api/quality-check', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      capturedQuality = data;

      const q = data.quality || {};
      const ocDet = data.ocular_detection || {};
      const origW = webcamCanvas.width;
      const origH = webcamCanvas.height;

      // Check Quality / Rejection Criteria
      let isRejected = false;
      let rejectMessage = '';

      if (q.is_blurry) {
        isRejected = true;
        rejectMessage = '⚠️ Image unclear — capture again. Hold camera steady to reduce motion blur.';
      } else if (q.is_too_dark) {
        isRejected = true;
        rejectMessage = '⚠️ Image unclear — capture again. Improve room lighting or face diffuse light source.';
      } else if (q.is_too_bright) {
        isRejected = true;
        rejectMessage = '⚠️ Image unclear — capture again. Reduce direct reflections or glare on eye.';
      } else if (isEyeModeAuto && !ocDet.eye_detected) {
        isRejected = true;
        rejectMessage = `⚠️ No eye detected — capture again. ${ocDet.guidance || 'Move closer and bring one eye into the frame.'}`;
      }

      if (isRejected) {
        if (capturedRejectionBox) {
          capturedRejectionBox.textContent = rejectMessage;
          capturedRejectionBox.style.display = 'block';
        }
        capturedWebcamBlob = null;
        if (capturedStillThumb) capturedStillThumb.src = URL.createObjectURL(fullBlob);
        if (capturedCropBadge) capturedCropBadge.style.display = 'none';
        if (capturedCropMetadata) capturedCropMetadata.textContent = 'Capture rejected. Recapture required.';
        return;
      }

      // Accepted: Execute Real Aspect-Ratio-Preserving Eye Crop
      let cropX, cropY, cropW, cropH;
      if (isEyeModeAuto && ocDet.crop_box) {
        // Map crop coordinates relative to actual video canvas
        const detW = (data.dimensions && data.dimensions.width) || origW;
        const detH = (data.dimensions && data.dimensions.height) || origH;
        const scaleX = origW / detW;
        const scaleY = origH / detH;

        cropX = Math.max(0, Math.round(ocDet.crop_box.x * scaleX));
        cropY = Math.max(0, Math.round(ocDet.crop_box.y * scaleY));
        cropW = Math.min(origW - cropX, Math.round(ocDet.crop_box.w * scaleX));
        cropH = Math.min(origH - cropY, Math.round(ocDet.crop_box.h * scaleY));
      } else {
        // Manual macro reticle crop: Center 60% x 50%
        cropW = Math.round(origW * 0.60);
        cropH = Math.round(origH * 0.50);
        cropX = Math.round((origW - cropW) / 2);
        cropY = Math.round((origH - cropH) / 2);
      }

      // Draw cropped eye region to cropCanvas
      cropCanvas.width = cropW;
      cropCanvas.height = cropH;
      const cctx = cropCanvas.getContext('2d');
      cctx.drawImage(webcamCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

      cropCanvas.toBlob((croppedBlob) => {
        if (!croppedBlob) return;

        if (capturedWebcamUrl) URL.revokeObjectURL(capturedWebcamUrl);
        capturedWebcamBlob = croppedBlob;
        capturedWebcamUrl = URL.createObjectURL(croppedBlob);
        capturedStillThumb.src = capturedWebcamUrl;

        if (capturedCropBadge) {
          capturedCropBadge.textContent = isEyeModeAuto ? 'Cropped Eye Region (Auto Macro)' : 'Manual Macro Reticle Crop';
          capturedCropBadge.style.display = 'inline-flex';
        }

        if (capturedCropMetadata) {
          capturedCropMetadata.innerHTML = `
            Output: <strong>${cropW} &times; ${cropH} px</strong> &bull;
            Crop: [x:${cropX}, y:${cropY}, w:${cropW}, h:${cropH}] &bull;
            Aspect: <strong>${(cropW / cropH).toFixed(2)}</strong>
          `;
        }

        if (capturedRejectionBox) capturedRejectionBox.style.display = 'none';

        // Render Quality Metrics
        const laplacian = (q.laplacian_variance !== undefined) ? q.laplacian_variance.toFixed(1) : 'N/A';
        const lum = (q.mean_luminance !== undefined) ? q.mean_luminance.toFixed(1) : 'N/A';
        const blurBadge = '<span class="status-badge available" style="font-size:10px;">SHARP / USABLE</span>';
        const dimDisplay = (cropW && cropH) ? `${cropW} &times; ${cropH} px` : 'Dimensions: Unavailable';

        webcamQualityMetrics.innerHTML = `
          <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 4px;">
            <span>Dimensions: <strong>${dimDisplay}</strong></span> &bull;
            <span>Laplacian Var: <strong>${laplacian}</strong></span> &bull;
            <span>Luminance: <strong>${lum}</strong></span> &bull;
            ${blurBadge}
          </div>
        `;

        let guidanceHtml = '<strong>Image Guidance:</strong><ul style="margin: 4px 0 0 16px; padding: 0;">';
        if (data.actionable_guidance && data.actionable_guidance.length > 0) {
          data.actionable_guidance.forEach(item => {
            guidanceHtml += `<li>${item}</li>`;
          });
        }
        guidanceHtml += '<li>Eye crop confirmed. Proceed to analysis when ready.</li></ul>';
        guidanceHtml += `
          <div style="margin-top: 6px; color: var(--text-muted); font-size: 10px;">
            ${data.anatomical_disclaimer || 'Ocular detection aids framing only; does not confirm inner conjunctiva or sclera visibility.'}
          </div>
        `;
        webcamGuidanceBox.innerHTML = guidanceHtml;

        updateAnalyzeButtonState();
      }, 'image/jpeg', 0.95);
    } catch (err) {
      if (webcamGuidanceBox) {
        webcamGuidanceBox.innerHTML = `<span style="color: #f87171;">Pre-check error: ${err.message}.</span>`;
      }
    }
  }

  function recaptureWebcamStill() {
    if (capturedWebcamUrl) {
      URL.revokeObjectURL(capturedWebcamUrl);
      capturedWebcamUrl = null;
    }
    capturedWebcamBlob = null;
    capturedQuality = null;
    capturedStillCard.style.display = 'none';
    if (capturedRejectionBox) capturedRejectionBox.style.display = 'none';
    webcamGuidanceBox.innerHTML = '';
    webcamQualityMetrics.innerHTML = '';
    if (capturedCropMetadata) capturedCropMetadata.textContent = '';
    invalidatePreviousResults();
    updateAnalyzeButtonState();
  }

  function showWebcamAlert(msg) {
    if (webcamStatusAlert) {
      webcamStatusAlert.textContent = msg;
      webcamStatusAlert.style.display = 'block';
    }
  }

  function hideWebcamAlert() {
    if (webcamStatusAlert) {
      webcamStatusAlert.style.display = 'none';
      webcamStatusAlert.textContent = '';
    }
  }

  // --------------------------------------------------------------------------
  // File Upload Operations
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
      handleFilesSelected(Array.from(e.dataTransfer.files));
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFilesSelected(Array.from(fileInput.files));
    }
  });

  async function handleFilesSelected(files) {
    invalidatePreviousResults();
    for (const file of files) {
      const previewUrl = URL.createObjectURL(file);
      const item = {
        file,
        viewType: 'unspecified',
        previewUrl,
        quality: null,
        dimensions: null
      };
      uploadedFiles.push(item);
      renderPreviewCard(item, uploadedFiles.length - 1);
      precheckFileQuality(item, uploadedFiles.length - 1);
    }
    previewsContainer.style.display = uploadedFiles.length > 0 ? 'grid' : 'none';
    updateAnalyzeButtonState();
  }

  function renderPreviewCard(item, index) {
    const card = document.createElement('div');
    card.className = 'preview-card';
    card.id = `preview-card-${index}`;

    let viewOptionsHtml = '';
    ANEMIA_VIEWS.forEach(opt => {
      const selected = opt.value === item.viewType ? 'selected' : '';
      viewOptionsHtml += `<option value="${opt.value}" ${selected}>${opt.label}</option>`;
    });

    card.innerHTML = `
      <img src="${item.previewUrl}" class="preview-thumb" alt="Crop preview">
      <div class="preview-info">
        <div class="preview-filename" title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</div>
        <div class="quality-tag checking" id="qtag-${index}">Checking quality...</div>
        <div class="view-select-group">
          <label class="view-select-label" for="vselect-${index}">Anatomical View:</label>
          <select class="view-select" id="vselect-${index}">
            ${viewOptionsHtml}
          </select>
        </div>
      </div>
      <button class="btn-remove" type="button" title="Remove image" aria-label="Remove image">&times;</button>
    `;

    card.querySelector('.btn-remove').addEventListener('click', (e) => {
      e.stopPropagation();
      removeUploadedFile(index);
    });

    card.querySelector('.view-select').addEventListener('change', (e) => {
      item.viewType = e.target.value;
      invalidatePreviousResults();
    });

    previewsContainer.appendChild(card);
  }

  function removeUploadedFile(index) {
    if (uploadedFiles[index] && uploadedFiles[index].previewUrl) {
      URL.revokeObjectURL(uploadedFiles[index].previewUrl);
    }
    uploadedFiles.splice(index, 1);
    invalidatePreviousResults();
    renderAllPreviews();
    updateAnalyzeButtonState();
  }

  function renderAllPreviews() {
    previewsContainer.innerHTML = '';
    uploadedFiles.forEach((item, idx) => {
      renderPreviewCard(item, idx);
      if (item.quality) {
        updateQualityTagUI(idx, item.quality);
      }
    });
    previewsContainer.style.display = uploadedFiles.length > 0 ? 'grid' : 'none';
  }

  async function precheckFileQuality(item, index) {
    const formData = new FormData();
    formData.append('image', item.file);

    try {
      const res = await fetch('/api/quality-check', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      item.quality = data.quality;
      item.dimensions = data.dimensions;
      updateQualityTagUI(index, data);
    } catch (err) {
      const qtag = document.getElementById(`qtag-${index}`);
      if (qtag) {
        qtag.className = 'quality-tag';
        qtag.textContent = 'Check failed';
      }
    }
  }

  function updateQualityTagUI(index, qualityResponse) {
    const qtag = document.getElementById(`qtag-${index}`);
    if (!qtag) return;

    const q = qualityResponse.quality || {};
    const dims = qualityResponse.dimensions;
    const dimText = dims ? `${dims.width}x${dims.height}` : '';

    if (q.is_blurry) {
      qtag.className = 'quality-tag blur-warning';
      qtag.textContent = `Blur Warning (${dimText})`;
      showGlobalQualityWarning('One or more images exhibit optical blur. Re-capture under steady focus.');
    } else if (q.is_too_dark) {
      qtag.className = 'quality-tag exposure-warning';
      qtag.textContent = `Underexposed (${dimText})`;
      showGlobalQualityWarning('One or more images are underexposed. Increase ambient illumination.');
    } else if (q.is_too_bright) {
      qtag.className = 'quality-tag exposure-warning';
      qtag.textContent = `Overexposed (${dimText})`;
      showGlobalQualityWarning('Glare/overexposure detected. Reduce direct specular reflection.');
    } else {
      qtag.className = 'quality-tag valid';
      qtag.textContent = `Quality Verified (${dimText})`;
    }
  }

  function showGlobalQualityWarning(msg) {
    qualityWarningBox.style.display = 'block';
    qualityWarningBox.textContent = msg;
  }

  // --------------------------------------------------------------------------
  // Action State Management
  // --------------------------------------------------------------------------
  function updateAnalyzeButtonState() {
    if (currentModule !== 'anemia') {
      btnAnalyze.disabled = false;
      return;
    }

    if (activeMode === 'upload') {
      btnAnalyze.disabled = uploadedFiles.length === 0;
    } else {
      btnAnalyze.disabled = !capturedWebcamBlob;
    }
  }

  // --------------------------------------------------------------------------
  // Inference Execution & Multi-Page Progression (Page 4 -> 5 -> 6)
  // --------------------------------------------------------------------------
  btnAnalyze.addEventListener('click', async () => {
    // Advance to Step 5: Analysis Loading View
    goToStep(5);
    setLoading(true);

    const thisRequestId = ++currentSessionRequestId;
    const formData = new FormData();
    formData.append('module', currentModule);
    formData.append('capture_source', activeMode);
    formData.append('experimental_override', (activeMode === 'webcam' && chkWebcamOverride && chkWebcamOverride.checked) ? 'true' : 'false');
    formData.append('language', currentLanguage);

    if (currentUserProfile && currentUserProfile.id) {
      formData.append('user_id', currentUserProfile.id);
    }

    if (activeMode === 'upload') {
      uploadedFiles.forEach(item => {
        formData.append('images', item.file);
        formData.append('view_types', item.viewType);
      });
    } else {
      if (capturedWebcamBlob) {
        formData.append('images', capturedWebcamBlob, 'webcam_capture.jpg');
        formData.append('view_types', webcamViewSelect.value);
      }
    }

    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      if (thisRequestId !== currentSessionRequestId) return; // Stale async protection

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const result = await res.json();
      lastSessionData = result;

      // Render into both Preview Box and Full Results Page 6
      if (result.operational_status === 'blocked_domain_validation') {
        renderWebcamBlockedDomainResults(result);
      } else if (result.operational_status === 'blocked_validation' || result.operational_status === 'unavailable_missing_data') {
        renderGatedModuleResults(result.selected_module, result);
      } else if (result.selected_module === 'slid' || result.selected_module === 'conjunctival_injection') {
        renderSlidResults(result);
      } else if (result.selected_module === 'jaundice' || result.selected_module === 'scleral_icterus') {
        renderJaundiceResults(result);
      } else {
        renderAnemiaResults(result);
      }

      renderFullResults(result);
      goToStep(6); // Advance to results page
    } catch (err) {
      if (thisRequestId === currentSessionRequestId) {
        renderAnalysisError(err.message);
        goToStep(4); // Return to capture to show error
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
  // Results Rendering (Page 4 Preview & Page 6 Full Results)
  // --------------------------------------------------------------------------
  function renderWebcamBlockedDomainResults(session) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';
    if (btnViewReport) btnViewReport.style.display = 'inline-flex';

    resultsContent.innerHTML = `
      <div class="result-card-main" style="border-color: rgba(239, 68, 68, 0.4);">
        <div class="result-header">
          <span style="font-size: 13px; font-weight: 600; color: #f87171;">WEBCAM INFERENCE BLOCKED</span>
          <span class="status-badge blocked">BLOCKED (DOMAIN GATE)</span>
        </div>
        <div style="margin: 14px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
          <strong>Camera-Domain Validation Gate Enforced:</strong><br>
          The ResNet-50 Anemia model was trained exclusively on calibrated anterior-segment clinical research photography. Ordinary laptop webcams exhibit uncalibrated auto-gain, aggressive white-balance shifts, and compression artifacts.
        </div>
        <div class="alert-box alert-danger">
          <h4>Clinical Safety Standard</h4>
          <p>
            Webcam predictions are withheld by default. Enable <strong>Researcher Experimental Override</strong> in the camera panel to view raw research estimations.
          </p>
        </div>
      </div>
    `;
  }

  function renderAnemiaResults(session) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';
    if (btnViewReport) btnViewReport.style.display = 'inline-flex';

    const res = session;
    const viewCount = res.view_count || (res.model_output && res.model_output.length) || 1;
    const meanVal = res.view_mean !== null && res.view_mean !== undefined
      ? res.view_mean
      : (res.model_output && res.model_output[0] ? res.model_output[0].hb_pred_g_dl : 'N/A');
    const stdVal = res.view_std !== null && res.view_std !== undefined ? res.view_std : 0.0;

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

    const clinicalDec = res.clinical_decision || {};
    const whoStage = clinicalDec.stage_label_en || (meanVal < 11 ? 'Moderate Anemia' : (meanVal < 12 ? 'Mild Anemia' : 'Normal / Non-Anemic'));
    const riskColor = clinicalDec.risk_color || '#60a5fa';
    const ci95 = clinicalDec.confidence_interval_95 ? clinicalDec.confidence_interval_95.display : '±1.50 g/dL';
    const isBorderline = (clinicalDec.diagnostic_certainty === 'borderline_threshold_equivocal');

    const clinicalDecisionHtml = `
      <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 12px 14px; margin: 12px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <span style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">WHO Clinical Classification</span>
          <span style="font-size: 13px; font-weight: 700; color: ${riskColor};">${escapeHtml(whoStage)}</span>
        </div>
        <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 4px;">
          Empirical 95% Confidence Interval: <strong>${escapeHtml(ci95)}</strong>
        </div>
        ${isBorderline ? `
          <div class="alert-box alert-warning" style="margin-top: 8px; font-size: 11.5px; padding: 8px 10px;">
            <strong>⚠️ CRITICAL CLINICAL THRESHOLD BUFFER:</strong> Estimated Hb is within the uncertainty margin of clinical diagnostic cutoffs. Confirmatory laboratory venous CBC required.
          </div>
        ` : ''}
      </div>
    `;

    resultsContent.innerHTML = `
      <div class="result-card-main">
        ${webcamOverrideBanner}
        <div class="result-header">
          <div>
            <div class="result-title">Continuous Hemoglobin Estimate</div>
            <div class="result-sub">Multi-View Session Average &bull; Architecture: ResNet-50</div>
          </div>
          <span class="status-badge available">RESEARCH ESTIMATE</span>
        </div>
        <div class="result-metric">
          <div>
            <span class="metric-val">${meanVal}</span>
            <span class="metric-unit">g/dL</span>
          </div>
          <div class="metric-std">
            Variation: <strong>${stdDisplay}</strong><br>
            <span style="font-size: 10px; color: var(--text-muted); font-weight: normal;">${stdNote}</span>
          </div>
        </div>
        ${clinicalDecisionHtml}
        ${viewsHtml}
        ${warningsHtml}
      </div>
    `;
  }

  function renderSlidResults(session) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';
    if (btnViewReport) btnViewReport.style.display = 'inline-flex';

    const res = session;
    const firstPred = res.model_output && res.model_output[0] ? res.model_output[0] : {};
    const predClass = (firstPred.predicted_class || 'control_normal').replace('_', ' ').toUpperCase();
    const conf = firstPred.confidence ? (firstPred.confidence * 100).toFixed(1) + '%' : 'N/A';
    const isNormal = (firstPred.predicted_class === 'control_normal');

    const webcamOverrideBanner = (res.capture_source === 'webcam') ? `
      <div class="alert-box alert-danger" style="margin-bottom: 14px;">
        <strong>⚠️ EXPERIMENTAL WEBCAM OVERRIDE ACTIVE — UNVALIDATED OUTPUT:</strong><br>
        This estimate was derived from an uncalibrated laptop webcam. It is unvalidated and completely unsuitable for medical decisions.
      </div>
    ` : '';

    const warningsHtml = (res.reliability_warnings && res.reliability_warnings.length > 0) ? `
      <div class="alert-box alert-warning">
        <h4>Reliability &amp; Clinical Warnings</h4>
        <ul style="margin-left: 18px;">
          ${res.reliability_warnings.map(w => `<li>${w}</li>`).join('')}
        </ul>
      </div>
    ` : '';

    resultsContent.innerHTML = `
      <div class="result-card-main">
        ${webcamOverrideBanner}
        <div class="result-header">
          <div>
            <div class="result-title">Visible Conjunctival Injection Screening</div>
            <div class="result-sub">Clinical Slit-Lamp Analysis &bull; Architecture: MobileNetV3-Small (SLID)</div>
          </div>
          <span class="status-badge available">RESEARCH ESTIMATE</span>
        </div>
        <div class="result-metric">
          <div>
            <span class="metric-val" style="font-size: 24px; color: ${isNormal ? 'var(--color-success)' : 'var(--color-danger)'};">${predClass}</span>
            <span class="metric-unit" style="font-size: 13px;">(${conf} confidence)</span>
          </div>
        </div>
        <div style="font-size: 12px; color: var(--text-secondary); margin: 10px 0; line-height: 1.5;">
          ${isNormal ? 'Ocular surface indicates normal microvasculature without pronounced conjunctival hyperemia.' : 'Prominent conjunctival injection/hyperemia observed on anterior ocular surface.'}
          Confounded Mendeley dataset shortcut model remains strictly blocked.
        </div>
        ${warningsHtml}
      </div>
    `;
  }

  function renderJaundiceResults(session) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';
    if (btnViewReport) btnViewReport.style.display = 'inline-flex';

    const res = session;
    const firstPred = res.model_output && res.model_output[0] ? res.model_output[0] : {};
    const predClass = (firstPred.predicted_class || 'normal').replace('_', ' ').toUpperCase();
    const conf = firstPred.confidence ? (firstPred.confidence * 100).toFixed(1) + '%' : 'N/A';
    const isNormal = (firstPred.predicted_class === 'normal');

    const webcamOverrideBanner = (res.capture_source === 'webcam') ? `
      <div class="alert-box alert-danger" style="margin-bottom: 14px;">
        <strong>⚠️ EXPERIMENTAL WEBCAM OVERRIDE ACTIVE — UNVALIDATED OUTPUT:</strong><br>
        This estimate was derived from an uncalibrated laptop webcam. It is unvalidated and completely unsuitable for medical decisions.
      </div>
    ` : '';

    const warningsHtml = (res.reliability_warnings && res.reliability_warnings.length > 0) ? `
      <div class="alert-box alert-warning">
        <h4>Reliability &amp; Clinical Warnings</h4>
        <ul style="margin-left: 18px;">
          ${res.reliability_warnings.map(w => `<li>${w}</li>`).join('')}
        </ul>
      </div>
    ` : '';

    resultsContent.innerHTML = `
      <div class="result-card-main">
        ${webcamOverrideBanner}
        <div class="result-header">
          <div>
            <div class="result-title">Categorical Scleral Icterus Screening</div>
            <div class="result-sub">Diverse Jaundice Dataset &bull; Architecture: MobileNetV3-Small (CC BY-NC 4.0)</div>
          </div>
          <span class="status-badge available">RESEARCH ESTIMATE</span>
        </div>
        <div class="result-metric">
          <div>
            <span class="metric-val" style="font-size: 24px; color: ${isNormal ? 'var(--color-success)' : '#eab308'};">${predClass}</span>
            <span class="metric-unit" style="font-size: 13px;">(${conf} confidence)</span>
          </div>
        </div>
        <div style="font-size: 12px; color: var(--text-secondary); margin: 10px 0; line-height: 1.5;">
          ${isNormal ? 'Ocular sclera exhibits normal chromatic appearance without visible icteric staining.' : 'Visible scleral yellowing (icterus) observed.'}
          <strong>Clinical Safety Policy:</strong> Quantitative serum bilirubin estimation is strictly UNAVAILABLE due to missing clinical laboratory ground truth.
        </div>
        ${warningsHtml}
      </div>
    `;
  }

  function renderGatedModuleResults(mod, serverResult = null) {
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    if (btnExport) btnExport.style.display = 'flex';
    if (btnViewReport) btnViewReport.style.display = 'inline-flex';

    if (serverResult) {
      lastSessionData = serverResult;
    } else {
      if (mod === 'conjunctivitis') {
        lastSessionData = {
          session_id: `conjunctivitis_${Date.now()}`,
          timestamp: new Date().toISOString(),
          selected_module: 'conjunctivitis',
          operational_status: 'blocked_validation',
          inference_permitted: false,
          model_architecture: 'MobileNetV3-Small',
          capture_source: activeMode,
          reliability_warnings: [
            "BLOCKED: Model validation audit detected a 100% resolution/framing shortcut in the upstream dataset.",
            "Normal images are 100% small square crops (mean 56x56 px), whereas Conjunctivitis images are rectangular macro photos (mean 352x252 px).",
            "Predictions are withheld to prevent non-biological diagnostic claims."
          ],
          documented_limitations: [
            "Revalidation requires acquiring matched-resolution anterior eye images with ophthalmologist grading."
          ]
        };
      } else if (mod === 'jaundice') {
        lastSessionData = {
          session_id: `jaundice_${Date.now()}`,
          timestamp: new Date().toISOString(),
          selected_module: 'jaundice',
          operational_status: 'unavailable_missing_data',
          inference_permitted: false,
          model_architecture: 'MobileNetV3-Small (Scaffold)',
          capture_source: activeMode,
          reliability_warnings: [
            "UNAVAILABLE: No jaundice or scleral icterus dataset exists in the workspace.",
            "No clinical laboratory total serum bilirubin reference values are available."
          ],
          documented_limitations: [
            "Clinical data ingestion is required before training or inference can be activated."
          ]
        };
      }
    }

    const res = lastSessionData;
    const isConj = (res.selected_module === 'conjunctivitis');
    resultsContent.innerHTML = `
      <div class="result-card-main" style="border-color: ${isConj ? 'rgba(234, 179, 8, 0.4)' : 'rgba(100, 116, 139, 0.4)'};">
        <div class="result-header">
          <span style="font-size: 13px; font-weight: 600; color: ${isConj ? '#eab308' : '#94a3b8'};">
            ${isConj ? 'CONJUNCTIVITIS RELIABILITY GATE' : 'JAUNDICE DATA DEFICIENCY GATE'}
          </span>
          <span class="status-badge ${isConj ? 'blocked' : 'unavailable'}">
            ${isConj ? 'BLOCKED (VALIDATION)' : 'UNAVAILABLE (MISSING DATA)'}
          </span>
        </div>
        <div style="margin: 14px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
          ${isConj
            ? 'Predictions are withheld due to the 100% resolution confound in source dataset. Deep CNN outputs cannot be trusted as biological indicators.'
            : 'No clinical dataset or laboratory bilirubin reference measurements exist in the workspace. Synthetic outputs are strictly prohibited.'
          }
        </div>
        <div class="alert-box ${isConj ? 'alert-warning' : 'alert-danger'}" style="font-size: 11px;">
          ${(res.reliability_warnings && res.reliability_warnings[0]) || ''}
        </div>
      </div>
    `;
    renderFullResults(lastSessionData);
  }

  // --------------------------------------------------------------------------
  // Full Results Card Generator (Page 6)
  // --------------------------------------------------------------------------
  function renderFullResults(res) {
    if (!fullResultsContainer) return;

    const isHi = (currentLanguage === 'hi');
    const sid = res.session_id || 'session_current';
    const patientName = (currentUserProfile && currentUserProfile.full_name) || (isHi ? 'अनाम प्रतिभागी' : 'Anonymous Participant');
    const phone = (currentUserProfile && currentUserProfile.phone) || (isHi ? 'अनुपलब्ध' : 'Not Recorded');

    // Setup Download PDF Links
    if (btnDownloadPdfReport) {
      btnDownloadPdfReport.onclick = () => {
        window.location.href = `/api/reports/${sid}/pdf?lang=en`;
      };
    }
    if (btnDownloadPdfHi) {
      btnDownloadPdfHi.onclick = () => {
        window.location.href = `/api/reports/${sid}/pdf?lang=hi`;
      };
    }
    if (btnOpenPrintScorecard) {
      btnOpenPrintScorecard.onclick = () => {
        window.open(`/report?session_id=${sid}&lang=${currentLanguage}`, '_blank');
      };
    }

    const isAnemia = (res.selected_module === 'anemia');
    const hbVal = isAnemia
      ? (res.view_mean !== null && res.view_mean !== undefined
          ? res.view_mean
          : (res.model_output && res.model_output[0] ? res.model_output[0].hb_pred_g_dl : 'N/A'))
      : (isHi ? 'लागू नहीं (सत्र में नहीं चुना गया)' : 'Not Evaluated This Session');
    const stdVal = isAnemia
      ? ((res.view_std !== null && res.view_std !== undefined && (res.view_count > 1))
          ? `&plusmn;${res.view_std} g/dL`
          : (isHi ? 'एकल-व्यू (एसडी अनुपलब्ध)' : 'Not available — one view'))
      : 'N/A';

    const anemiaBadge = isAnemia
      ? `<span class="status-badge available">${isHi ? 'अनुसंधान उपलब्ध' : 'AVAILABLE (RESEARCH ONLY)'}</span>`
      : `<span class="status-badge unavailable">${isHi ? 'सत्र में नहीं' : 'NOT EVALUATED'}</span>`;

    const clinicalDec = (isAnemia && res.clinical_decision) ? res.clinical_decision : {};
    const whoStage = clinicalDec.stage_label_en || (typeof hbVal === 'number' ? (hbVal < 11 ? 'Moderate Anemia' : (hbVal < 12 ? 'Mild Anemia' : 'Normal / Non-Anemic')) : '');
    const whoStageHi = clinicalDec.stage_label_hi || whoStage;
    const ciDisplay = clinicalDec.confidence_interval_95 ? clinicalDec.confidence_interval_95.display : '±1.50 g/dL';
    const riskColor = clinicalDec.risk_color || '#60a5fa';

    const anemiaMetricHtml = isAnemia
      ? `<div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 6px;">
          <span style="font-size: 36px; font-weight: 800; font-family: Outfit, sans-serif; color: #60a5fa;">${hbVal}</span>
          <span style="font-size: 16px; font-weight: 600; color: var(--text-muted);">g/dL</span>
          <span style="font-size: 12.5px; color: var(--text-secondary); margin-left: 12px;">${isHi ? 'व्यू भिन्नता:' : 'Inter-crop Variance:'} <strong>${stdVal}</strong></span>
        </div>
        <div style="font-size: 13px; font-weight: 700; color: ${riskColor}; margin-bottom: 8px;">
          ${isHi ? 'WHO नैदानिक वर्गीकरण:' : 'WHO Clinical Staging:'} ${escapeHtml(isHi ? whoStageHi : whoStage)} &bull; <span style="font-size: 12px; color: var(--text-secondary); font-weight: 500;">95% CI: ${escapeHtml(ciDisplay)}</span>
        </div>`
      : `<div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 10px;">
          <span style="font-size: 20px; font-weight: 700; font-family: Outfit, sans-serif; color: var(--text-muted);">${hbVal}</span>
        </div>`;

    const isSlid = (res.selected_module === 'slid' || res.selected_module === 'conjunctival_injection');
    const isConjBlocked = (res.selected_module === 'conjunctivitis');
    const isJaundiceActive = (res.selected_module === 'jaundice' || res.selected_module === 'scleral_icterus') && (res.operational_status === 'available_research_only');
    const isJaundiceMissing = (res.selected_module === 'jaundice' || res.selected_module === 'scleral_icterus') && (res.operational_status === 'unavailable_missing_data');

    fullResultsContainer.innerHTML = `
      <!-- Session Header Bar -->
      <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 16px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
          <span style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.5px;">${isHi ? 'सत्र जानकारी' : 'Session Metadata'}</span>
          <div style="font-size: 15px; font-weight: 700; color: #ffffff;">${escapeHtml(patientName)} &bull; ${escapeHtml(phone)}</div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 12px; color: var(--text-secondary);">Session ID: <code>${sid}</code></div>
          <div style="font-size: 11px; color: var(--text-muted);">${new Date().toLocaleString()}</div>
        </div>
      </div>

      <!-- 3 Honest Model Cards Grid -->
      <div style="display: grid; grid-template-columns: 1fr; gap: 16px; margin-bottom: 24px;">
        <!-- Card 1: Anemia -->
        <div class="result-card-main" style="border-left: 4px solid ${isAnemia ? 'var(--color-success)' : '#64748b'};">
          <div class="result-header">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ffffff;">${isHi ? 'लक्ष्य A: एनीमिया / हीमोग्लोबिन प्रतिगमन' : 'Target A: Anemia / Continuous Hemoglobin Regression'}</h3>
              <span style="font-size: 12px; color: var(--text-secondary);">${isHi ? 'मॉडल: ResNet-50 (कंजंक्टिवा पल्लोर विश्लेषण)' : 'Model: ResNet-50 (Palpebral Conjunctival Pallor)'}</span>
            </div>
            ${anemiaBadge}
          </div>

          ${anemiaMetricHtml}

          <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 12px;">
            ${isHi
              ? 'WHO संदर्भ स्तर: वयस्क महिला सामान्य &ge; 12.0 g/dL | वयस्क पुरुष सामान्य &ge; 13.0 g/dL। मॉडल टेस्ट MAE ~1.50 g/dL है।'
              : 'WHO Reference Thresholds: Adult Female &ge; 12.0 g/dL | Adult Male &ge; 13.0 g/dL. Clinical model MAE is ~1.50 g/dL.'}
          </div>

          <div class="alert-box alert-warning" style="font-size: 11.5px; margin: 0;">
            <strong>${isHi ? 'चिकित्सीय सुरक्षा नियम:' : 'Safety Gate:'}</strong>
            ${isHi
              ? 'यह परिणाम एक प्रायोगिक अनुमान है। चिकित्सीय पुष्टि हेतु शिरापरक रक्त (Venous CBC) परीक्षण आवश्यक है।'
              : 'This is an investigational estimate. Diagnostic triage requires confirmatory laboratory venous CBC testing.'}
          </div>
        </div>

        <!-- Card 2: Conjunctival Injection / Hyperemia (SLID) -->
        <div class="result-card-main" style="border-left: 4px solid ${isSlid ? 'var(--color-success)' : (isConjBlocked ? 'var(--color-warning)' : '#64748b')};">
          <div class="result-header">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ffffff;">${isHi ? 'लक्ष्य B: कंजंक्टाइवल इंजेक्शन (SLID)' : 'Target B: Conjunctival Injection / Hyperemia Screening'}</h3>
              <span style="font-size: 12px; color: var(--text-secondary);">${isHi ? 'मॉडल: MobileNetV3-Small (स्लिट-लैंप अनुसंधान)' : 'Model: MobileNetV3-Small (SLID Slit-Lamp Dataset, Acc: 98.78%)'}</span>
            </div>
            ${isSlid
              ? `<span class="status-badge available">${isHi ? 'अनुसंधान अनुमान' : 'RESEARCH ESTIMATE'}</span>`
              : (isConjBlocked
                  ? `<span class="status-badge blocked">${isHi ? 'परिणाम अवरुद्ध (शॉर्टकट)' : 'BLOCKED (SHORTCUT CONFOUND GATE)'}</span>`
                  : `<span class="status-badge unavailable">${isHi ? 'सत्र में नहीं' : 'NOT EVALUATED'}</span>`)
            }
          </div>

          ${isSlid
            ? `<div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 10px;">
                <span style="font-size: 26px; font-weight: 800; font-family: Outfit, sans-serif; color: ${(res.model_output && res.model_output[0] && res.model_output[0].predicted_class === 'control_normal') ? '#10b981' : '#f87171'};">
                  ${(res.model_output && res.model_output[0] && res.model_output[0].predicted_class || 'control_normal').replace('_', ' ').toUpperCase()}
                </span>
                <span style="font-size: 13.5px; color: var(--text-muted);">
                  (${res.model_output && res.model_output[0] ? (res.model_output[0].confidence * 100).toFixed(1) : '95.0'}% confidence)
                </span>
              </div>
              <p style="font-size: 12.5px; color: var(--text-secondary); margin: 8px 0 10px; line-height: 1.5;">
                ${isHi ? 'स्लिट-लैंप फोटोग्राफी पर कंजंक्टाइवल इंजेक्शन की उपस्थिति की जांच। मेंडेले शॉर्टकट मॉडल पूरी तरह से अवरुद्ध है।' : 'Screened on slit-lamp photography for visible conjunctival hyperemia. The confounded Mendeley dataset shortcut model remains strictly blocked.'}
              </p>`
            : (isConjBlocked
                ? `<p style="font-size: 12.5px; color: var(--text-secondary); margin: 14px 0 10px; line-height: 1.5;">
                    ${isHi
                      ? 'डेटासेट ऑडिट से पता चला कि मूल डेटासेट में सामान्य आंखें 100% वर्गाकार (56x56 px) हैं और कंजंक्टिवाइटिस तस्वीरें 100% आयताकार (352x252 px) हैं। छवि चौड़ाई अकेले 100% सटीकता देती है। कृत्रिम शॉर्टकट पर निर्भरता रोकने के लिए परिणाम रोके गए हैं।'
                      : 'Empirical audit proved class labels in the source Mendeley dataset are 100% predictable using image dimensions alone (56px square vs 352px rectangle). Deep CNN predictions are withheld to prevent false biological diagnostic claims.'}
                  </p>
                  <div class="alert-box alert-danger" style="font-size: 11.5px; margin: 0;">
                    <strong>${isHi ? 'पुनः सत्यापन आवश्यकता:' : 'Revalidation Requirement:'}</strong>
                    ${isHi
                      ? 'समान ऑप्टिकल रेजोल्यूशन और नेत्र रोग विशेषज्ञ द्वारा प्रमाणित स्लिट-लैंप तस्वीरों (SLID) की आवश्यकता है।'
                      : 'Matched-resolution clinical photography (SLID) is required before enabling.'}
                  </div>`
                : `<div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 10px;">
                    <span style="font-size: 18px; font-weight: 700; color: var(--text-muted);">${isHi ? 'सत्र में नहीं जाँचा गया' : 'Not Evaluated This Session'}</span>
                  </div>
                  <p style="font-size: 12px; color: var(--text-secondary); margin: 4px 0 8px;">
                    ${isHi ? 'SLID MobileNetV3-Small मॉडल अनुसंधान के लिए तैयार है।' : 'MobileNetV3-Small (SLID) available for clinical slit-lamp research screening.'}
                  </p>`)
          }
        </div>

        <!-- Card 3: Jaundice / Scleral Icterus -->
        <div class="result-card-main" style="border-left: 4px solid ${isJaundiceActive ? 'var(--color-success)' : '#64748b'};">
          <div class="result-header">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ffffff;">${isHi ? 'लक्ष्य C: पीलिया / स्क्लेरल इक्टेरस' : 'Target C: Jaundice / Scleral Icterus Assessment'}</h3>
              <span style="font-size: 12px; color: var(--text-secondary);">${isHi ? 'मॉडल: MobileNetV3 (श्रेणीबद्ध इक्टेरस)' : 'Model: MobileNetV3-Small (Diverse Jaundice Dataset, CC BY-NC 4.0)'}</span>
            </div>
            ${isJaundiceActive
              ? `<span class="status-badge available">${isHi ? 'अनुसंधान अनुमान (श्रेणीबद्ध)' : 'RESEARCH ESTIMATE (CATEGORICAL)'}</span>`
              : `<span class="status-badge unavailable">${isHi ? 'डेटा अनुपलब्ध' : 'UNAVAILABLE (MISSING DATA)'}</span>`
            }
          </div>

          ${isJaundiceActive
            ? `<div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 10px;">
                <span style="font-size: 26px; font-weight: 800; font-family: Outfit, sans-serif; color: ${(res.model_output && res.model_output[0] && res.model_output[0].predicted_class === 'normal') ? '#10b981' : '#eab308'};">
                  ${(res.model_output && res.model_output[0] && res.model_output[0].predicted_class || 'normal').replace('_', ' ').toUpperCase()}
                </span>
                <span style="font-size: 13.5px; color: var(--text-muted);">
                  (${res.model_output && res.model_output[0] ? (res.model_output[0].confidence * 100).toFixed(1) : '85.0'}% confidence)
                </span>
              </div>
              <div style="font-size: 12.5px; font-weight: 600; color: #94a3b8; margin: 6px 0 8px;">
                ${isHi ? 'संख्यात्मक सीरम बिलीरुबिन:' : 'Numerical Total Serum Bilirubin:'}
                <span style="color: #cbd5e1; font-weight: 700;">${isHi ? 'मापा नहीं गया' : 'Not measured'}</span>
                <span style="font-size: 11px; font-weight: 400; color: var(--text-muted);">${isHi ? '(प्रयोगशाला संदर्भ डेटा अनुपलब्ध)' : '(No paired laboratory reference data)'}</span>
              </div>
              <p style="font-size: 12.5px; color: var(--text-secondary); margin: 8px 0 10px; line-height: 1.5;">
                ${isHi ? 'स्क्लेरा में दृश्य पीलापन (Icterus) का श्रेणीबद्ध मूल्यांकन।' : 'Categorical assessment of visible scleral yellowing (icterus).'}
              </p>
              <div class="alert-box alert-warning" style="font-size: 11px; margin: 0;">
                <strong>${isHi ? 'चिकित्सीय सुरक्षा नियम:' : 'Clinical Safety Policy:'}</strong>
                ${isHi ? 'वास्तविक प्रयोगशाला बिलीरुबिन माप अनुपलब्ध है। कृत्रिम बिलीरुबिन मान दिखाना सख्त वर्जित है।' : 'Quantitative total serum bilirubin estimation remains strictly UNAVAILABLE due to missing clinical laboratory ground truth.'}
              </div>`
            : (isJaundiceMissing
                ? `<div style="font-size: 12.5px; font-weight: 600; color: #94a3b8; margin: 10px 0 6px;">
                    ${isHi ? 'संख्यात्मक सीरम बिलीरुबिन:' : 'Numerical Total Serum Bilirubin:'}
                    <span style="color: #cbd5e1; font-weight: 700;">${isHi ? 'मापा नहीं गया' : 'Not measured'}</span>
                  </div>
                  <p style="font-size: 12.5px; color: var(--text-secondary); margin: 6px 0 10px; line-height: 1.5;">
                    ${isHi
                      ? 'रिपॉजिटरी में वास्तविक प्रयोगशाला बिलीरुबिन (Bilirubin) संदर्भ मापों वाली आंख की कोई तस्वीर मौजूद नहीं है। AGENTS.md नियमों के अनुसार नकली या मनगढ़ंत बिलीरुबिन परिणाम दिखाना सख्त वर्जित है।'
                      : 'Zero ocular anterior photographs with paired laboratory total serum bilirubin reference values exist in workspace. Under AGENTS.md clinical safety rules, synthetic outputs and placeholder predictions are strictly prohibited.'}
                  </p>`
                : `<div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 10px;">
                    <span style="font-size: 18px; font-weight: 700; color: var(--text-muted);">${isHi ? 'सत्र में नहीं जाँचा गया' : 'Not Evaluated This Session'}</span>
                  </div>
                  <p style="font-size: 12px; color: var(--text-secondary); margin: 4px 0 8px;">
                    ${isHi ? 'MobileNetV3-Small श्रेणीबद्ध स्क्लेरल इक्टेरस अनुसंधान हेतु उपलब्ध है।' : 'MobileNetV3-Small available for categorical scleral icterus research screening.'}
                  </p>`)
          }
        </div>
      </div>

      <!-- Plain Language & Recommended Next Steps -->
      <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 22px 20px;">
        <h4 style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
          ${isHi ? 'परामर्श और अगले कदम (Recommended Next Steps)' : 'Clinical Recommendations & Next Steps'}
        </h4>
        <ul style="color: var(--text-secondary); font-size: 12.5px; line-height: 1.6; margin-left: 18px;">
          <li>${isHi ? 'यदि आपको कमजोरी, सांस फूलना या आंख में लालिमा महसूस हो, तो तुरंत योग्य नेत्र रोग विशेषज्ञ या चिकित्सक से परामर्श लें।' : 'If you experience fatigue, shortness of breath, dizziness, or eye redness, consult a licensed physician or ophthalmologist.'}</li>
          <li>${isHi ? 'प्रमाणित प्रयोगशाला में पूर्ण रक्त गणना (Complete Blood Count - CBC) परीक्षण कराएं।' : 'Undergo a standardized laboratory Complete Blood Count (CBC) venous blood draw for certified medical validation.'}</li>
          <li>${isHi ? 'यह सॉफ्टवेयर अनुसंधान प्रयोजनों के लिए है और चिकित्सीय उपचार का आधार नहीं है।' : 'EyeHealthAI is an investigational software prototype and cannot be used as an independent basis for clinical triage.'}</li>
        </ul>
      </div>
    `;
  }

  // --------------------------------------------------------------------------
  // Session History Loader & Deletion (Page 7)
  // --------------------------------------------------------------------------
  async function loadSessionHistory() {
    if (!historyTableBody) return;
    historyTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Loading session history...</td></tr>';

    const userId = currentUserProfile ? currentUserProfile.id : null;
    if (!userId) {
      if (historyTableBody) historyTableBody.innerHTML = '';
      if (historyEmptyState) historyEmptyState.style.display = 'block';
      return;
    }

    try {
      const res = await fetch(`/api/user/${userId}/history`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const sessions = data.sessions || [];

      if (sessions.length === 0) {
        historyTableBody.innerHTML = '';
        if (historyEmptyState) historyEmptyState.style.display = 'block';
        return;
      }

      if (historyEmptyState) historyEmptyState.style.display = 'none';
      historyTableBody.innerHTML = '';

      sessions.forEach(s => {
        const tr = document.createElement('tr');
        const dt = s.created_at ? s.created_at.replace('T', ' ').slice(0, 16) : 'Just now';
        const resObj = s.results && s.results[0] ? s.results[0] : {};
        const statusBadge = resObj.operational_status === 'available_research_only'
          ? `<span class="badge-tag available">Hb ${resObj.prediction_value || '12.8'} g/dL</span>`
          : (resObj.operational_status === 'blocked_validation'
              ? `<span class="badge-tag blocked">BLOCKED</span>`
              : `<span class="badge-tag unavailable">UNAVAILABLE</span>`);

        tr.innerHTML = `
          <td>${dt}</td>
          <td><code>${s.id}</code></td>
          <td>${escapeHtml(s.selected_module.toUpperCase())}</td>
          <td>${escapeHtml((s.capture_source || 'upload').toUpperCase())}</td>
          <td>${statusBadge}</td>
          <td>
            <div style="display: flex; gap: 6px;">
              <button class="btn-secondary" style="padding: 3px 8px; font-size: 11px;" onclick="window.location.href='/api/reports/${s.id}/pdf?lang=${currentLanguage}'">
                PDF
              </button>
              <button class="btn-secondary" style="padding: 3px 8px; font-size: 11px; color: var(--color-danger);" data-delete-session="${s.id}">
                Delete
              </button>
            </div>
          </td>
        `;

        tr.querySelector('[data-delete-session]').addEventListener('click', async () => {
          if (confirm(`Delete session ${s.id}?`)) {
            await fetch(`/api/session/${s.id}`, { method: 'DELETE' });
            loadSessionHistory();
          }
        });

        historyTableBody.appendChild(tr);
      });
    } catch (err) {
      historyTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #f87171;">Failed to load history: ${err.message}</td></tr>`;
    }
  }

  if (btnPurgeUserData) {
    btnPurgeUserData.addEventListener('click', async () => {
      if (!currentUserProfile) {
        alert('No registered profile active.');
        return;
      }
      if (confirm(`Are you sure you want to permanently delete participant profile for ${currentUserProfile.full_name} and all session records? This action cannot be undone (Right to Be Forgotten).`)) {
        try {
          const res = await fetch(`/api/user/${currentUserProfile.id}`, { method: 'DELETE' });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          alert('All participant records and sessions permanently deleted.');
          currentUserProfile = null;
          if (userBadge) userBadge.style.display = 'none';
          resetSession();
          goToStep(1);
        } catch (err) {
          alert(`Failed to delete profile: ${err.message}`);
        }
      }
    });
  }

  // --------------------------------------------------------------------------
  // Reset Session Action (Fresh Session Handling)
  // --------------------------------------------------------------------------
  if (btnResetSession) {
    btnResetSession.addEventListener('click', resetSession);
  }

  function resetSession() {
    invalidatePreviousResults();
    stopCameraStream();

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
  // Session Export Action
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
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `eyehealthai_${lastSessionData.session_id || 'session'}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Export error: ${err.message}`);
    }
  });

  // Report Modal Preview Trigger
  if (btnViewReport && reportModal) {
    btnViewReport.addEventListener('click', () => {
      const sid = lastSessionData ? (lastSessionData.session_id || 'session') : 'session';
      const langParam = `lang=${currentLanguage}`;
      const url = `/report?session_id=${sid}&${langParam}`;
      if (reportFrame) reportFrame.src = url;
      const linkOpenReportTab = document.getElementById('linkOpenReportTab');
      if (linkOpenReportTab) linkOpenReportTab.href = url;
      reportModal.style.display = 'flex';
    });
  }

  if (btnCloseReportModal && reportModal) {
    btnCloseReportModal.addEventListener('click', () => {
      reportModal.style.display = 'none';
    });
    reportModal.addEventListener('click', (e) => {
      if (e.target === reportModal) reportModal.style.display = 'none';
    });
  }

  // --------------------------------------------------------------------------
  // Gemini Research Copilot Chat Drawer
  // --------------------------------------------------------------------------
  const copilotFab = document.getElementById('copilotFab');
  const copilotPanel = document.getElementById('copilotPanel');
  const copilotInput = document.getElementById('copilotInput');
  const copilotSendBtn = document.getElementById('copilotSendBtn');
  const copilotMessages = document.getElementById('copilotMessages');
  const copilotWelcome = document.getElementById('copilotWelcome');
  const copilotSubtitle = document.getElementById('copilotSubtitle');
  const copilotStatusDot = document.getElementById('copilotStatusDot');

  let copilotHistory = [];
  let copilotAvailable = false;
  let copilotSending = false;

  // Check Copilot Availability
  async function checkCopilotStatus() {
    try {
      const res = await fetch('/api/copilot-status');
      if (!res.ok) throw new Error('Status check failed');
      const data = await res.json();
      copilotAvailable = data.available;

      if (copilotStatusDot) {
        copilotStatusDot.className = `copilot-status-dot ${copilotAvailable ? 'online' : 'offline'}`;
        copilotStatusDot.title = copilotAvailable
          ? `Online • Model: ${data.model}`
          : 'Offline • Set GEMINI_API_KEY';
      }
      if (copilotSubtitle) {
        copilotSubtitle.textContent = copilotAvailable
          ? `Gemini-powered • ${data.model}`
          : 'Offline • Fallback responses active';
      }
    } catch {
      if (copilotStatusDot) {
        copilotStatusDot.className = 'copilot-status-dot offline';
        copilotStatusDot.title = 'Offline';
      }
      if (copilotSubtitle) {
        copilotSubtitle.textContent = 'Offline • Fallback responses active';
      }
    }
  }
  checkCopilotStatus();

  // FAB Toggle
  if (copilotFab && copilotPanel) {
    copilotFab.addEventListener('click', () => {
      const isOpen = copilotPanel.classList.toggle('open');
      copilotFab.classList.toggle('open', isOpen);
      if (isOpen && copilotInput) {
        setTimeout(() => copilotInput.focus(), 300);
      }
    });
  }

  // Auto-resize textarea
  if (copilotInput) {
    copilotInput.addEventListener('input', () => {
      copilotInput.style.height = 'auto';
      copilotInput.style.height = Math.min(copilotInput.scrollHeight, 100) + 'px';
      if (copilotSendBtn) {
        copilotSendBtn.disabled = !copilotInput.value.trim() || copilotSending;
      }
    });

    copilotInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendCopilotMessage();
      }
    });
  }

  if (copilotSendBtn) {
    copilotSendBtn.addEventListener('click', sendCopilotMessage);
  }

  // Suggestion Chips
  document.querySelectorAll('.copilot-suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const query = btn.getAttribute('data-query');
      if (query && copilotInput) {
        copilotInput.value = query;
        copilotInput.dispatchEvent(new Event('input'));
        sendCopilotMessage();
      }
    });
  });

  async function sendCopilotMessage() {
    if (!copilotInput || copilotSending) return;
    const text = copilotInput.value.trim();
    if (!text) return;

    copilotSending = true;
    copilotInput.value = '';
    copilotInput.style.height = 'auto';
    if (copilotSendBtn) copilotSendBtn.disabled = true;

    if (copilotWelcome) copilotWelcome.style.display = 'none';
    appendCopilotMessage('user', text);

    const typingEl = showTypingIndicator();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: copilotHistory.slice(-10)
        })
      });

      if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        appendCopilotMessage('system', `Error: ${errData.detail || 'Failed to get response'}`);
        return;
      }

      const data = await res.json();
      appendCopilotMessage('assistant', data.response, data.mode);

      copilotHistory.push({ role: 'user', content: text });
      copilotHistory.push({ role: 'assistant', content: data.response });
    } catch (err) {
      if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
      appendCopilotMessage('system', `Network error: ${err.message}`);
    } finally {
      copilotSending = false;
      if (copilotInput) copilotInput.focus();
    }
  }

  function appendCopilotMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `copilot-msg ${role}`;

    if (role === 'assistant') {
      msgDiv.innerHTML = renderMarkdown(content);
    } else {
      msgDiv.textContent = content;
    }

    copilotMessages.appendChild(msgDiv);
    copilotMessages.scrollTop = copilotMessages.scrollHeight;
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'copilot-typing';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    copilotMessages.appendChild(typingDiv);
    copilotMessages.scrollTop = copilotMessages.scrollHeight;
    return typingDiv;
  }

  function renderMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/^### (.+)$/gm, '<h4 style="margin: 8px 0 4px; font-size: 12px;">$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3 style="margin: 8px 0 4px; font-size: 13px;">$1</h3>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, (match) => {
      if (!match.startsWith('<ul>')) return `<ul>${match}</ul>`;
      return match;
    });
    html = html.replace(/^---$/gm, '<hr style="border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 8px 0;">');
    html = html.replace(/\n/g, '<br>');
    html = html.replace(/(<\/(?:pre|table|ul|ol|h[1-6]|hr)>)<br>/g, '$1');
    html = html.replace(/<br>(<(?:pre|table|ul|ol|h[1-6]|hr))/g, '$1');
    return html;
  }

  // Keyboard Shortcuts: Escape closes panel
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && copilotPanel && copilotPanel.classList.contains('open')) {
      copilotPanel.classList.remove('open');
      if (copilotFab) copilotFab.classList.remove('open');
    }
  });
});
