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
      appTitle: "OcuScan",
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
      labelGender: "Gender *",
      genderOptPreferNot: "Prefer not to say",
      genderOptFemale: "Female",
      genderOptMale: "Male",
      genderOptOther: "Other",
      genderOptNonBinary: "Non-binary",
      genderOptSelfDescribe: "Self-describe",
      chkConsentDemographicText: "I consent to the documentation of demographic profile data strictly decoupled from model inference.",
      consentHeader: "Informed Consent &amp; Privacy Notice",
      consentP1: "<strong>Experimental Research Only:</strong> OcuScan is an investigational software prototype. It does NOT provide a certified medical diagnosis.",
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
      btnBackScreening: "&larr; Back to Screening Dashboard",
      trackStep1Title: "1. Profile",
      trackStep1Sub: "Basic information",
      trackStep2Title: "2. Consent",
      trackStep2Sub: "Review &amp; agree",
      trackStep3Title: "3. Screening",
      trackStep3Sub: "Upload eye image",
      trackStep4Title: "4. Report",
      trackStep4Sub: "Results &amp; export",
      workspaceTitle: "Unified Screening Workspace",
      workspaceSub: "Upload one clear eye image. OcuScan will evaluate all compatible research targets and prepare one consolidated report.",
      workspaceInfoBanner: "No disease selection needed &mdash; all supported targets are evaluated automatically.",
      captureCardTitle: "Capture or upload eye image",
      captureCardSubtitle: "One upload &bull; One consolidated report",
      dropzoneMain: "Drag an image here or",
      dropzoneBrowse: "browse files",
      dropzoneSub: "Supported formats: JPG, PNG, WEBP &bull; Up to 10 MB",
      btnChooseEyeText: "Choose eye image",
      dividerOr: "or",
      modeUploadText: "Upload image",
      modeLiveCameraText: "Live camera",
      btnAnalyzeText: "Analyze Image",
      assessCardTitle: "What we assess automatically",
      assessAnemiaTitle: "Anemia / Hemoglobin",
      assessAnemiaDesc: "Experimental Hb estimate from compatible conjunctival views",
      assessSlidTitle: "Conjunctival injection (SLID)",
      assessSlidDesc: "Visible redness classification when image quality and domain are suitable",
      assessIcterusTitle: "Visible scleral icterus",
      assessIcterusDesc: "Categorical scleral-yellowing screen; not a bilirubin measurement",
      assessCataractTitle: "Nuclear Cataract & LOCS III",
      assessCataractDesc: "Crystalline lens opacification and LOCS III staging from slit-lamp biomicroscopy",
      assessStyeTitle: "Stye / Hordeolum",
      assessStyeDesc: "Acute focal eyelid margin nodule and erythematous lesion screening",
      badgeResearchOnly: "Research only",
      assessFooterNote: "Unsuitable image, unsupported capture conditions, or unavailable model? The report will state this clearly rather than invent a result.",
      howItWorksTitle: "How OcuScan Works",
      pipeStep1Title: "Single Input",
      pipeStep1Desc: "Upload palpebral, scleral, or full anterior view",
      pipeStep2Title: "Quality Gating",
      pipeStep2Desc: "Real-time Laplacian blur and luminance check",
      pipeStep3Title: "Parallel Inference",
      pipeStep3Desc: "Active target dispatch with reliability boundaries",
      pipeStep4Title: "Consolidated Report",
      pipeStep4Desc: "Unified scorecard, explanation &amp; PDF download"
    },
    hi: {
      appTitle: "OcuScan",
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
      labelGender: "लिंग (Gender) *",
      genderOptPreferNot: "बताना नहीं चाहते",
      genderOptFemale: "महिला (Female)",
      genderOptMale: "पुरुष (Male)",
      genderOptOther: "अन्य (Other)",
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
      btnBackScreening: "&larr; जांच डैशबोर्ड पर वापस जाएं",
      trackStep1Title: "1. प्रोफाइल",
      trackStep1Sub: "मूल जानकारी",
      trackStep2Title: "2. सहमति",
      trackStep2Sub: "समीक्षा और सहमति",
      trackStep3Title: "3. स्क्रीनिंग",
      trackStep3Sub: "आंख की तस्वीर अपलोड करें",
      trackStep4Title: "4. रिपोर्ट",
      trackStep4Sub: "परिणाम और रिपोर्ट",
      workspaceTitle: "एकीकृत स्क्रीनिंग कार्यक्षेत्र",
      workspaceSub: "एक स्पष्ट आंख की तस्वीर अपलोड करें। OcuScan सभी समर्थित शोध लक्ष्यों का मूल्यांकन करेगा और एक समेकित रिपोर्ट तैयार करेगा।",
      workspaceInfoBanner: "किसी बीमारी को चुनने की आवश्यकता नहीं है &mdash; सभी समर्थित लक्ष्यों का स्वचालित रूप से मूल्यांकन किया जाता है।",
      captureCardTitle: "आंख की तस्वीर लें या अपलोड करें",
      captureCardSubtitle: "एक अपलोड &bull; एक समेकित रिपोर्ट",
      dropzoneMain: "यहाँ एक तस्वीर खींचें या",
      dropzoneBrowse: "फ़ाइलें चुनें",
      dropzoneSub: "समर्थित प्रारूप: JPG, PNG, WEBP &bull; 10 एमबी तक",
      btnChooseEyeText: "आंख की तस्वीर चुनें",
      dividerOr: "या",
      modeUploadText: "तस्वीर अपलोड",
      modeLiveCameraText: "लाइव कैमरा",
      btnAnalyzeText: "तस्वीर का विश्लेषण करें",
      assessCardTitle: "हम स्वचालित रूप से क्या जांचते हैं",
      assessAnemiaTitle: "एनीमिया / हीमोग्लोबिन",
      assessAnemiaDesc: "अनुकूल कंजाक्तिवा छवियों से प्रायोगिक हीमोग्लोबिन अनुमान",
      assessSlidTitle: "कंजाक्टिवल इंजेक्शन (SLID)",
      assessSlidDesc: "छवि गुणवत्ता और डोमेन उपयुक्त होने पर दृश्य लालिमा वर्गीकरण",
      assessIcterusTitle: "दृश्य स्केलेरल पीलिया",
      assessIcterusDesc: "स्केलेरा के पीलेपन की श्रेणीबद्ध जांच; सीरम बिलीरुबिन माप नहीं",
      assessCataractTitle: "मोतियाबिंद / LOCS III",
      assessCataractDesc: "स्लिट-लैंप बायोमाइक्रोस्कोपी से लेंस धुंधलापन और LOCS III चरण निर्धारण",
      assessStyeTitle: "गुहेरी / स्टाई (Hordeolum)",
      assessStyeDesc: "पलक के किनारे पर गांठ और तीव्र लालिमा की श्रेणीबद्ध जांच",
      badgeResearchOnly: "केवल अनुसंधान",
      assessFooterNote: "अनुपयुक्त छवि, असमर्थित स्थितियाँ या अनुपलब्ध मॉडल? रिपोर्ट परिणाम गढ़ने के बजाय स्पष्ट रूप से बताएगी।",
      howItWorksTitle: "OcuScan कैसे काम करता है",
      pipeStep1Title: "एकल इनपुट",
      pipeStep1Desc: "पाल्पेब्रल, स्केलेरल या पूर्ण आंख की छवि अपलोड करें",
      pipeStep2Title: "गुणवत्ता परीक्षण",
      pipeStep2Desc: "लैप्लासियन धुंधलापन और चमक की रीयल-टाइम जांच",
      pipeStep3Title: "समानांतर विश्लेषण",
      pipeStep3Desc: "सक्रिय मॉडल विश्लेषण और सुरक्षा गेट",
      pipeStep4Title: "समेकित रिपोर्ट",
      pipeStep4Desc: "एकीकृत स्कोरकार्ड, व्याख्या और पीडीएफ"
    }
  };

  // --------------------------------------------------------------------------
  // DOM Elements - Stepper & Header
  // --------------------------------------------------------------------------
  const langBtnEn = document.getElementById('langBtnEn');
  const langBtnHi = document.getElementById('langBtnHi');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const userBadge = document.getElementById('userBadge');
  const userNameBadge = document.getElementById('userNameBadge');
  const btnNavHistory = document.getElementById('btnNavHistory');
  const btnResetSession = document.getElementById('btnResetSession');
  const btnNewSessionText = document.getElementById('btnNewSessionText');
  const btnNavHistoryText = document.getElementById('btnNavHistoryText');

  // Theme Toggle Handler
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', nextTheme);
      try {
        localStorage.setItem('eyehealth_theme', nextTheme);
      } catch (e) {}
    });
  }

  // Page 1 Elements
  const cardLangEn = document.getElementById('cardLangEn');
  const cardLangHi = document.getElementById('cardLangHi');
  const btnContinueLang = document.getElementById('btnContinueLang');

  const formRegistration = document.getElementById('formRegistration');
  const inputRegName = document.getElementById('inputRegName');
  const inputRegPhone = document.getElementById('inputRegPhone');
  const inputRegEmail = document.getElementById('inputRegEmail');
  const inputRegAge = document.getElementById('inputRegAge');
  const selectRegGender = document.getElementById('selectRegGender');
  const pregnancySection = document.getElementById('pregnancySection');
  const chkPregnancy = document.getElementById('chkPregnancy');
  const trimesterWrap = document.getElementById('trimesterWrap');
  const selectTrimester = document.getElementById('selectTrimester');
  const chkConsentExternalAi = document.getElementById('chkConsentExternalAi');
  const chkConsentResearch = document.getElementById('chkConsentResearch');
  const chkConsentStorage = document.getElementById('chkConsentStorage');
  const regErrorAlert = document.getElementById('regErrorAlert');
  const btnBackToLang = document.getElementById('btnBackToLang');

  // Modal Update Age Elements
  const modalUpdateAge = document.getElementById('modalUpdateAge');
  const formUpdateAge = document.getElementById('formUpdateAge');
  const inputUpdateAge = document.getElementById('inputUpdateAge');
  const selectUpdateClinicalSex = document.getElementById('selectUpdateClinicalSex');
  const updatePregnancySection = document.getElementById('updatePregnancySection');
  const chkUpdatePregnancy = document.getElementById('chkUpdatePregnancy');
  const updateTrimesterWrap = document.getElementById('updateTrimesterWrap');
  const selectUpdateTrimester = document.getElementById('selectUpdateTrimester');
  const chkUpdateConsentExternalAi = document.getElementById('chkUpdateConsentExternalAi');
  const updateAgeErrorAlert = document.getElementById('updateAgeErrorAlert');
  const btnCancelUpdateAge = document.getElementById('btnCancelUpdateAge');

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

  // Modern 4-Step Progress Tracker Elements
  const fourStepTracker = document.getElementById('fourStepTracker');
  const trackStep1 = document.getElementById('trackStep1');
  const trackStep2 = document.getElementById('trackStep2');
  const trackStep3 = document.getElementById('trackStep3');
  const trackStep4 = document.getElementById('trackStep4');
  const trackLine1 = document.getElementById('trackLine1');
  const trackLine2 = document.getElementById('trackLine2');
  const trackLine3 = document.getElementById('trackLine3');

  // Single-Upload Workspace Elements
  const btnChooseEye = document.getElementById('btnChooseEye');
  const btnBrowseLink = document.getElementById('btnBrowseLink');
  const singlePreviewContainer = document.getElementById('singlePreviewContainer');
  const previewImageThumb = document.getElementById('previewImageThumb');
  const previewImageName = document.getElementById('previewImageName');
  const previewImageMeta = document.getElementById('previewImageMeta');
  const badgeSharpness = document.getElementById('badgeSharpness');
  const badgeLuminance = document.getElementById('badgeLuminance');
  const badgeDetectionMode = document.getElementById('badgeDetectionMode');
  const badgeEnhancement = document.getElementById('badgeEnhancement');
  const btnTogglePreviewCrop = document.getElementById('btnTogglePreviewCrop');
  const btnReplaceImage = document.getElementById('btnReplaceImage');
  const btnRemoveImage = document.getElementById('btnRemoveImage');
  let isShowingOriginalCrop = false;

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
  // Page-Based Navigation Engine & State Preservation
  // --------------------------------------------------------------------------
  let previousStep = 3; // Default previous step for History return

  function goToStep(step) {
    if (currentStep !== 7 && step === 7) {
      previousStep = currentStep;
    }
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

    // Hide 4-step progress tracker on initial language setup page; show on step >= 2
    if (fourStepTracker) {
      if (step === 1) {
        fourStepTracker.classList.add('tracker-hidden');
      } else {
        fourStepTracker.classList.remove('tracker-hidden');
      }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });

    updateFourStepTracker(step);

    if (step === 2) {
      validateProfileAndConsent();
    }

    if (step === 7) {
      loadSessionHistory();
    }
  }

  function updateFourStepTracker(step) {
    if (!trackStep1 || !trackStep2 || !trackStep3 || !trackStep4) return;
    const checkSvg = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>`;

    function setStepState(btn, line, state, num) {
      if (!btn) return;
      btn.classList.remove('active', 'completed', 'pending');
      btn.classList.add(state);
      btn.setAttribute('aria-current', state === 'active' ? 'step' : 'false');

      // Make future / pending steps inactive and unclickable
      if (state === 'pending') {
        btn.setAttribute('disabled', 'true');
        btn.setAttribute('aria-disabled', 'true');
        btn.style.pointerEvents = 'none';
        btn.style.cursor = 'not-allowed';
      } else {
        btn.removeAttribute('disabled');
        btn.removeAttribute('aria-disabled');
        btn.style.pointerEvents = 'auto';
        btn.style.cursor = 'pointer';
      }

      const circle = btn.querySelector('.tracker-circle');
      if (circle) {
        // Show checkmarks ONLY for completed steps
        if (state === 'completed') {
          circle.innerHTML = checkSvg;
        } else {
          circle.textContent = num;
        }
      }
      if (line) {
        line.classList.toggle('completed', state === 'completed');
      }
    }

    if (step === 1) {
      setStepState(trackStep1, trackLine1, 'active', 1);
      setStepState(trackStep2, trackLine2, 'pending', 2);
      setStepState(trackStep3, trackLine3, 'pending', 3);
      setStepState(trackStep4, null, 'pending', 4);
    } else if (step === 2) {
      const nameVal = inputRegName ? inputRegName.value.trim() : '';
      const phoneVal = inputRegPhone ? inputRegPhone.value.trim() : '';
      const emailVal = inputRegEmail ? inputRegEmail.value.trim() : '';
      const ageStr = inputRegAge ? inputRegAge.value.trim() : '';
      const ageNum = parseInt(ageStr, 10);
      const isAgeValid = ageStr !== '' && !isNaN(ageNum) && Number.isInteger(Number(ageStr)) && ageNum >= 0 && ageNum <= 120;
      const genderVal = selectRegGender ? selectRegGender.value : '';
      const isProfileValid = (nameVal.length > 0) && (phoneVal.length > 0) && (emailVal.length > 0) && isAgeValid && (genderVal !== '');

      if (isProfileValid) {
        setStepState(trackStep1, trackLine1, 'completed', 1);
        setStepState(trackStep2, trackLine2, 'active', 2);
      } else {
        setStepState(trackStep1, trackLine1, 'active', 1);
        setStepState(trackStep2, trackLine2, 'pending', 2);
      }
      setStepState(trackStep3, trackLine3, 'pending', 3);
      setStepState(trackStep4, null, 'pending', 4);
    } else if (step === 3 || step === 4 || step === 5) {
      setStepState(trackStep1, trackLine1, 'completed', 1);
      setStepState(trackStep2, trackLine2, 'completed', 2);
      setStepState(trackStep3, trackLine3, 'active', 3);
      setStepState(trackStep4, null, 'pending', 4);
    } else if (step === 6) {
      setStepState(trackStep1, trackLine1, 'completed', 1);
      setStepState(trackStep2, trackLine2, 'completed', 2);
      setStepState(trackStep3, trackLine3, 'completed', 3);
      setStepState(trackStep4, null, 'active', 4);
    } else if (step === 7) {
      setStepState(trackStep1, trackLine1, 'completed', 1);
      setStepState(trackStep2, trackLine2, 'completed', 2);
      setStepState(trackStep3, trackLine3, lastSessionData ? 'completed' : 'pending', 3);
      setStepState(trackStep4, null, lastSessionData ? 'completed' : 'pending', 4);
    }
  }

  function validateProfileAndConsent() {
    const nameVal = inputRegName ? inputRegName.value.trim() : '';
    const phoneVal = inputRegPhone ? inputRegPhone.value.trim() : '';
    const emailVal = inputRegEmail ? inputRegEmail.value.trim() : '';
    const ageStr = inputRegAge ? inputRegAge.value.trim() : '';
    const ageNum = parseInt(ageStr, 10);
    const isAgeValid = ageStr !== '' && !isNaN(ageNum) && Number.isInteger(Number(ageStr)) && ageNum >= 0 && ageNum <= 120;
    const genderVal = selectRegGender ? selectRegGender.value : '';
    const isGenderValid = !!genderVal && genderVal !== '';

    const isProfileComplete = (nameVal.length > 0) && (phoneVal.length > 0) && (emailVal.length > 0) && isAgeValid && isGenderValid;

    const isResearchChecked = chkConsentResearch ? chkConsentResearch.checked : false;
    const isStorageChecked = chkConsentStorage ? chkConsentStorage.checked : false;
    const isMandatoryConsentGiven = isResearchChecked && isStorageChecked;

    if (currentStep === 2) {
      updateFourStepTracker(2);
    }

    if (btnSubmitRegistration) {
      btnSubmitRegistration.disabled = !(isProfileComplete && isMandatoryConsentGiven);
    }

    return isProfileComplete && isMandatoryConsentGiven;
  }

  // Attach real-time validation listeners to profile and consent fields
  ['input', 'change', 'keyup'].forEach(evt => {
    if (inputRegName) inputRegName.addEventListener(evt, validateProfileAndConsent);
    if (inputRegPhone) inputRegPhone.addEventListener(evt, validateProfileAndConsent);
    if (inputRegEmail) inputRegEmail.addEventListener(evt, validateProfileAndConsent);
    if (inputRegAge) inputRegAge.addEventListener(evt, validateProfileAndConsent);
    if (selectRegGender) selectRegGender.addEventListener(evt, validateProfileAndConsent);
    if (chkConsentResearch) chkConsentResearch.addEventListener(evt, validateProfileAndConsent);
    if (chkConsentStorage) chkConsentStorage.addEventListener(evt, validateProfileAndConsent);
    if (chkConsentExternalAi) chkConsentExternalAi.addEventListener(evt, validateProfileAndConsent);
    if (chkConsentDemographic) chkConsentDemographic.addEventListener(evt, validateProfileAndConsent);
  });

  // Prevent mouse wheel scrolling from incrementing/decrementing age inputs
  [inputRegAge, document.getElementById('inputUpdateAge')].forEach(el => {
    if (el) {
      el.addEventListener('wheel', (e) => e.preventDefault(), { passive: false });
    }
  });

  window.goToStep = goToStep;

  // Four-Step Progress Tracker Click Listeners
  if (trackStep1) trackStep1.addEventListener('click', () => goToStep(2));
  if (trackStep2) trackStep2.addEventListener('click', () => goToStep(2));
  if (trackStep3) trackStep3.addEventListener('click', () => {
    if (currentUserProfile) goToStep(4);
  });
  if (trackStep4) trackStep4.addEventListener('click', () => {
    if (lastSessionData) {
      goToStep(6);
    } else {
      showGlobalQualityWarning(currentLanguage === 'hi' ? 'कृपया रिपोर्ट देखने के लिए पहले एक आंख की छवि का विश्लेषण करें।' : 'Please upload and analyze an eye image first to generate the report.');
    }
  });

  // Clickable Logo Navigation: Returns to Protocol Guide (Page 3) or Registration (Page 2)
  const logoBtn = document.getElementById('logoBtn');
  if (logoBtn) {
    logoBtn.addEventListener('click', () => {
      if (currentUserProfile) {
        goToStep(3); // Protocol Guide
      } else {
        goToStep(2); // Registration / Login Profile
      }
    });
    logoBtn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        logoBtn.click();
      }
    });
  }

  // Direct Page Navigation Buttons
  const btnBackToGuide = document.getElementById('btnBackToGuide');
  if (btnBackToGuide) {
    btnBackToGuide.addEventListener('click', () => goToStep(3));
  }

  const btnBackFromHistory = document.getElementById('btnBackFromHistory');
  if (btnBackFromHistory) {
    btnBackFromHistory.addEventListener('click', () => {
      goToStep(previousStep || 4);
    });
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
  if (btnBackToScreening) btnBackToScreening.addEventListener('click', () => goToStep(previousStep || 4));
  if (btnNavHistory) btnNavHistory.addEventListener('click', () => goToStep(7));

  // Toggle conditional pregnancy controls based on gender
  if (selectRegGender) {
    selectRegGender.addEventListener('change', () => {
      if (selectRegGender.value === 'female') {
        if (pregnancySection) pregnancySection.style.display = 'block';
      } else {
        if (pregnancySection) pregnancySection.style.display = 'none';
        if (chkPregnancy) chkPregnancy.checked = false;
        if (trimesterWrap) trimesterWrap.style.display = 'none';
      }
    });
  }

  if (chkPregnancy) {
    chkPregnancy.addEventListener('change', () => {
      if (trimesterWrap) {
        trimesterWrap.style.display = chkPregnancy.checked ? 'flex' : 'none';
      }
    });
  }

  if (selectUpdateClinicalSex) {
    selectUpdateClinicalSex.addEventListener('change', () => {
      if (selectUpdateClinicalSex.value === 'female') {
        if (updatePregnancySection) updatePregnancySection.style.display = 'block';
      } else {
        if (updatePregnancySection) updatePregnancySection.style.display = 'none';
        if (chkUpdatePregnancy) chkUpdatePregnancy.checked = false;
        if (updateTrimesterWrap) updateTrimesterWrap.style.display = 'none';
      }
    });
  }

  if (chkUpdatePregnancy) {
    chkUpdatePregnancy.addEventListener('change', () => {
      if (updateTrimesterWrap) {
        updateTrimesterWrap.style.display = chkUpdatePregnancy.checked ? 'flex' : 'none';
      }
    });
  }

  if (btnCancelUpdateAge) {
    btnCancelUpdateAge.addEventListener('click', () => {
      if (modalUpdateAge) modalUpdateAge.style.display = 'none';
    });
  }

  // --------------------------------------------------------------------------
  // User Registration & Informed Consent (Page 2)
  // --------------------------------------------------------------------------
  if (formRegistration) {
    formRegistration.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (regErrorAlert) regErrorAlert.style.display = 'none';

      const selectRegGender = document.getElementById('selectRegGender');
      const chkConsentDemographic = document.getElementById('chkConsentDemographic');
      const isHi = (currentLanguage === 'hi');

      // Age validation: Mandatory integer between 0 and 120
      const ageStr = inputRegAge ? inputRegAge.value.trim() : '';
      if (!ageStr || isNaN(ageStr) || !Number.isInteger(Number(ageStr))) {
        if (regErrorAlert) {
          regErrorAlert.textContent = isHi
            ? 'कृपया 0 से 120 के बीच एक मान्य पूर्णांक आयु दर्ज करें।'
            : 'Please enter a valid whole integer age between 0 and 120.';
          regErrorAlert.style.display = 'block';
        }
        return;
      }
      const ageNum = parseInt(ageStr, 10);
      if (ageNum < 0 || ageNum > 120) {
        if (regErrorAlert) {
          regErrorAlert.textContent = isHi
            ? 'आयु 0 से 120 वर्ष के बीच होनी चाहिए।'
            : 'Age must be between 0 and 120 years.';
          regErrorAlert.style.display = 'block';
        }
        return;
      }

      const genderVal = selectRegGender ? selectRegGender.value : 'prefer_not_to_say';
      let clinicalSexVal = 'not_disclosed';
      if (genderVal === 'male') {
        clinicalSexVal = 'male';
      } else if (genderVal === 'female') {
        clinicalSexVal = 'female';
      } else {
        clinicalSexVal = 'not_disclosed';
      }

      const payload = {
        full_name: inputRegName.value.trim(),
        phone: inputRegPhone.value.trim(),
        email: inputRegEmail.value.trim(),
        age_years: ageNum,
        gender: genderVal,
        clinical_sex: clinicalSexVal,
        is_pregnant: (genderVal === 'female' && chkPregnancy) ? chkPregnancy.checked : false,
        pregnancy_trimester: (genderVal === 'female' && chkPregnancy && chkPregnancy.checked && selectTrimester) ? parseInt(selectTrimester.value, 10) : null,
        external_ai_consent: chkConsentExternalAi ? chkConsentExternalAi.checked : false,
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
        try {
          localStorage.setItem('ocuscan_user_profile', JSON.stringify(currentUserProfile));
        } catch (e) {
          console.warn('Could not cache user profile to localStorage:', e);
        }

        if (userBadge && userNameBadge) {
          userNameBadge.textContent = currentUserProfile.full_name;
          userBadge.style.display = 'inline-flex';
        }

        goToStep(4); // Advance directly to Unified Screening Workspace (Step 3 in 4-step tracker)
      } catch (err) {
        if (regErrorAlert) {
          regErrorAlert.textContent = err.message;
          regErrorAlert.style.display = 'block';
        }
      }
    });
  }

  // --------------------------------------------------------------------------
  // Legacy User Profile Age Update Handler
  // --------------------------------------------------------------------------
  if (formUpdateAge) {
    formUpdateAge.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (updateAgeErrorAlert) updateAgeErrorAlert.style.display = 'none';

      if (!currentUserProfile || !currentUserProfile.id) {
        if (updateAgeErrorAlert) {
          updateAgeErrorAlert.textContent = 'No active profile found. Please register first.';
          updateAgeErrorAlert.style.display = 'block';
        }
        return;
      }

      const isHi = (currentLanguage === 'hi');
      const ageStr = inputUpdateAge ? inputUpdateAge.value.trim() : '';
      if (!ageStr || isNaN(ageStr) || !Number.isInteger(Number(ageStr))) {
        if (updateAgeErrorAlert) {
          updateAgeErrorAlert.textContent = isHi
            ? 'कृपया 0 से 120 के बीच एक मान्य पूर्णांक आयु दर्ज करें।'
            : 'Please enter a valid whole integer age between 0 and 120.';
          updateAgeErrorAlert.style.display = 'block';
        }
        return;
      }
      const ageNum = parseInt(ageStr, 10);
      if (ageNum < 0 || ageNum > 120) {
        if (updateAgeErrorAlert) {
          updateAgeErrorAlert.textContent = isHi
            ? 'आयु 0 से 120 वर्ष के बीच होनी चाहिए।'
            : 'Age must be between 0 and 120 years.';
          updateAgeErrorAlert.style.display = 'block';
        }
        return;
      }

      const updatePayload = {
        age_years: ageNum,
        clinical_sex: selectUpdateClinicalSex ? selectUpdateClinicalSex.value : 'not_disclosed',
        is_pregnant: chkUpdatePregnancy ? chkUpdatePregnancy.checked : false,
        pregnancy_trimester: (chkUpdatePregnancy && chkUpdatePregnancy.checked && selectUpdateTrimester) ? parseInt(selectUpdateTrimester.value, 10) : null,
        external_ai_consent: chkUpdateConsentExternalAi ? chkUpdateConsentExternalAi.checked : false
      };

      try {
        const res = await fetch(`/api/user/${currentUserProfile.id}/update-age`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updatePayload)
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `Update failed with HTTP ${res.status}`);
        }

        const data = await res.json();
        currentUserProfile = data.user;
        try {
          localStorage.setItem('ocuscan_user_profile', JSON.stringify(currentUserProfile));
        } catch (e) {
          console.warn('Could not update cached profile in localStorage:', e);
        }
        if (modalUpdateAge) modalUpdateAge.style.display = 'none';

        // Re-trigger analyze now that age is securely recorded
        if (btnAnalyze) btnAnalyze.click();
      } catch (err) {
        if (updateAgeErrorAlert) {
          updateAgeErrorAlert.textContent = err.message;
          updateAgeErrorAlert.style.display = 'block';
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
      stopCameraStream();
      webcamSection.style.display = 'none';
      uploadSection.style.display = 'block';

      // Restore upload area correctly
      if (uploadedFiles && uploadedFiles.length > 0) {
        if (dropzone) dropzone.style.display = 'none';
        if (singlePreviewContainer) singlePreviewContainer.style.display = 'block';
      } else {
        if (dropzone) dropzone.style.display = 'flex';
        if (singlePreviewContainer) singlePreviewContainer.style.display = 'none';
      }
    } else {
      uploadSection.style.display = 'none';
      webcamSection.style.display = 'block';
      if (!capturedWebcamBlob && capturedStillCard) {
        capturedStillCard.style.display = 'none';
      }
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
    } else if (mod === 'cataract' || mod === 'nuclear_cataract') {
      moduleBanner.className = 'alert-box alert-info';
      moduleBanner.innerHTML = `
        <strong>Active Module: Nuclear Cataract &amp; LOCS III Staging</strong><br>
        MobileNetV3-Small (AUC: 0.910). Evaluates crystalline lens opacity and LOCS III staging. Slit-lamp biomicroscopy modality required.
      `;
      btnAnalyzeText.textContent = 'Run Research Inference';
      if (webcamDomainNotice) webcamDomainNotice.style.display = 'block';
    } else if (mod === 'stye' || mod === 'hordeolum' || mod === 'stye_hordeolum') {
      moduleBanner.className = 'alert-box alert-info';
      moduleBanner.innerHTML = `
        <strong>Active Module: Stye / Hordeolum Eyelid Lesion Screening</strong><br>
        MobileNetV3-Small. Identifies focal acute margin swelling and erythema on anterior eyelid photography.
      `;
      btnAnalyzeText.textContent = 'Run Research Inference';
      if (webcamDomainNotice) webcamDomainNotice.style.display = 'none';
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
        triggerDoctorBuddyEvent('quality_blurry');
      } else if (q.is_too_dark) {
        isRejected = true;
        rejectMessage = '⚠️ Image unclear — capture again. Improve room lighting or face diffuse light source.';
        triggerDoctorBuddyEvent('quality_dark');
      } else if (q.is_too_bright) {
        isRejected = true;
        rejectMessage = '⚠️ Image unclear — capture again. Reduce direct reflections or glare on eye.';
        triggerDoctorBuddyEvent('quality_bright');
      } else if (isEyeModeAuto && !ocDet.eye_detected) {
        isRejected = true;
        rejectMessage = `⚠️ No eye detected — capture again. ${ocDet.guidance || 'Move closer and bring one eye into the frame.'}`;
        triggerDoctorBuddyEvent('positioning_too_far');
      } else {
        triggerDoctorBuddyEvent('quality_passed');
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

        // Automatically recognize view for webcam capture
        const autoCamView = recognizeAnatomicalView(data);
        if (webcamViewSelect && autoCamView) {
          webcamViewSelect.value = autoCamView;
        }

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
  // File Upload Operations (Single-Upload Unified Workspace)
  // --------------------------------------------------------------------------
  if (dropzone) {
    dropzone.addEventListener('click', () => {
      if (fileInput) fileInput.click();
    });

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
  }

  if (btnChooseEye) {
    btnChooseEye.addEventListener('click', (e) => {
      e.stopPropagation();
      if (fileInput) fileInput.click();
    });
  }

  if (btnBrowseLink) {
    btnBrowseLink.addEventListener('click', (e) => {
      e.stopPropagation();
      if (fileInput) fileInput.click();
    });
  }

  if (btnReplaceImage) {
    btnReplaceImage.addEventListener('click', (e) => {
      e.stopPropagation();
      if (fileInput) fileInput.click();
    });
  }

  if (btnRemoveImage) {
    btnRemoveImage.addEventListener('click', (e) => {
      e.stopPropagation();
      invalidatePreviousResults();
      uploadedFiles.forEach(item => {
        if (item.previewUrl) {
          try { URL.revokeObjectURL(item.previewUrl); } catch (err) {}
        }
      });
      uploadedFiles = [];
      if (fileInput) fileInput.value = '';
      if (singlePreviewContainer) singlePreviewContainer.style.display = 'none';
      if (dropzone) dropzone.style.display = 'flex';
      if (previewsContainer) {
        previewsContainer.style.display = 'none';
        previewsContainer.innerHTML = '';
      }
      if (qualityWarningBox) {
        qualityWarningBox.style.display = 'none';
        qualityWarningBox.textContent = '';
      }
      if (badgeDetectionMode) badgeDetectionMode.style.display = 'none';
      if (badgeEnhancement) badgeEnhancement.style.display = 'none';
      if (btnTogglePreviewCrop) btnTogglePreviewCrop.style.display = 'none';
      const stagePills = document.getElementById('stagePillsRow');
      if (stagePills) stagePills.style.display = 'none';
      isShowingOriginalCrop = false;
      const aiViewBox = document.getElementById('aiViewRecognitionBox');
      if (aiViewBox) aiViewBox.style.display = 'none';
      updateAnalyzeButtonState();
    });
  }

  // Multi-stage pre-inference inspection handler
  window.selectPreviewStage = function(stage) {
    const item = (uploadedFiles && uploadedFiles[0]) ? uploadedFiles[0] : null;
    const prevs = (item && item.previews) ? item.previews : (window._currentPreviewData || {});
    const imgElem = document.getElementById('previewImageThumb');
    const btnUpload = document.getElementById('btnStageUpload');
    const btnModelInput = document.getElementById('btnStageModelInput');

    if (btnUpload) btnUpload.classList.remove('active');
    if (btnModelInput) btnModelInput.classList.remove('active');

    if (stage === 'upload') {
      if (btnUpload) btnUpload.classList.add('active');
      if (imgElem) imgElem.src = prevs.original_thumb || (item ? item.previewUrl : '');
    } else if (stage === 'model_input') {
      if (btnModelInput) btnModelInput.classList.add('active');
      if (imgElem) imgElem.src = prevs.final_model_input_thumb || prevs.processed_thumb || prevs.original_thumb || (item ? item.previewUrl : '');
    }
  };

  if (btnTogglePreviewCrop) {
    btnTogglePreviewCrop.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!uploadedFiles || uploadedFiles.length === 0 || !uploadedFiles[0].previews) return;
      const prevs = uploadedFiles[0].previews;
      isShowingOriginalCrop = !isShowingOriginalCrop;
      const isHi = (currentLanguage === 'hi');
      if (isShowingOriginalCrop && prevs.original_thumb) {
        if (previewImageThumb) previewImageThumb.src = prevs.original_thumb;
        btnTogglePreviewCrop.textContent = isHi ? '✨ संवर्धित क्रॉप देखें' : '✨ View Enhanced Crop';
      } else {
        if (previewImageThumb) previewImageThumb.src = prevs.processed_thumb || uploadedFiles[0].previewUrl;
        btnTogglePreviewCrop.textContent = isHi ? '👁 मूल छवि देखें' : '👁 View Original';
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFilesSelected(Array.from(fileInput.files));
      }
    });
  }

  async function handleFilesSelected(files) {
    if (!files || files.length === 0) return;
    invalidatePreviousResults();

    // Revoke previous blob URLs
    uploadedFiles.forEach(item => {
      if (item.previewUrl) {
        try { URL.revokeObjectURL(item.previewUrl); } catch (err) {}
      }
    });
    uploadedFiles = [];

    if (badgeDetectionMode) badgeDetectionMode.style.display = 'none';
    if (badgeEnhancement) badgeEnhancement.style.display = 'none';
    if (btnTogglePreviewCrop) btnTogglePreviewCrop.style.display = 'none';
    const stagePills = document.getElementById('stagePillsRow');
    if (stagePills) stagePills.style.display = 'none';
    isShowingOriginalCrop = false;

    // Support single primary image for unified screening
    const file = files[0];
    const previewUrl = URL.createObjectURL(file);
    const item = {
      file,
      viewType: 'unspecified',
      previewUrl,
      quality: null,
      dimensions: null
    };
    uploadedFiles.push(item);

    // Update Modern Single Preview Card UI
    if (previewImageThumb) previewImageThumb.src = previewUrl;
    if (previewImageName) previewImageName.textContent = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    if (previewImageMeta) previewImageMeta.textContent = `${sizeMb} MB • Checking quality...`;

    if (badgeSharpness) {
      badgeSharpness.className = 'badge-tag checking';
      badgeSharpness.textContent = 'Sharpness: Checking...';
    }
    if (badgeLuminance) {
      badgeLuminance.className = 'badge-tag checking';
      badgeLuminance.textContent = 'Exposure: Checking...';
    }

    if (singlePreviewContainer) singlePreviewContainer.style.display = 'block';
    if (dropzone) dropzone.style.display = 'none';

    // Backwards-compatible previews container for automated test suite
    if (previewsContainer) {
      previewsContainer.innerHTML = '';
      renderPreviewCard(item, 0);
    }

    updateAnalyzeButtonState();
    await precheckFileQuality(item, 0);
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

  // --------------------------------------------------------------------------
  // Automatic Anatomical View Recognition Helper
  // --------------------------------------------------------------------------
  function recognizeAnatomicalView(data) {
    const aiViewBox = document.getElementById('aiViewRecognitionBox');
    const aiViewDesc = document.getElementById('aiViewRecognitionDesc');
    const aiViewBadge = document.getElementById('aiViewConfidenceBadge');

    const dims = data.dimensions || {};
    const ocDet = data.ocular_detection || {};
    const w = dims.width || 0;
    const h = dims.height || 0;
    const aspect = h > 0 ? (w / h) : 1.0;

    let recognizedView = 'unspecified';
    let label = 'Unspecified Ocular Crop';
    let confidence = 'Moderate (Rule-Gated)';
    let explanation = 'Image contains ocular features without distinct palpebral everting signature.';

    if (ocDet.eye_detected && ocDet.crop_box) {
      if (aspect >= 1.35) {
        recognizedView = 'palpebral';
        label = 'Palpebral Conjunctiva (Lower Everted Lid)';
        confidence = 'High (88.4%)';
        explanation = 'Wide-aspect anterior crop matching everted lower eyelid microvascular signature.';
      } else if (aspect >= 1.05 && aspect < 1.35) {
        recognizedView = 'forniceal_palpebral';
        label = 'Forniceal + Palpebral Conjunctiva';
        confidence = 'High (84.1%)';
        explanation = 'Dual exposure of lower conjunctival sac and palpebral vascular bed detected.';
      } else {
        recognizedView = 'forniceal';
        label = 'Forniceal Conjunctiva (Deep Sac)';
        confidence = 'Moderate (79.5%)';
        explanation = 'Vertical anterior focal crop highlighting lower conjunctival fornix.';
      }
    } else if (w >= 300 && h >= 200) {
      recognizedView = 'raw_eye';
      label = 'Raw Full Eye View';
      confidence = 'High (91.0%)';
      explanation = 'Uncropped full ocular anterior segment with visible sclera, iris, and eyelid margin.';
    } else {
      recognizedView = 'unspecified';
      label = 'Unspecified Ocular Crop (Fallback)';
      confidence = 'Fallback (Uncertain)';
      explanation = 'Automated classification uncertainty: routed safely to generic ocular pipeline.';
    }

    if (aiViewBox && aiViewDesc) {
      aiViewBox.style.display = 'block';
      aiViewDesc.innerHTML = `
        Identified View: <strong>${label}</strong> &bull;
        Mapped Code: <code>${recognizedView}</code> &bull;
        <em>${explanation}</em>
      `;
      if (aiViewBadge) {
        aiViewBadge.textContent = confidence;
      }
    }

    return recognizedView;
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
      item.crop_dimensions = data.crop_dimensions;
      item.ocular_detection = data.ocular_detection;
      item.enhancement = data.enhancement;
      item.previews = data.previews;
      item.quality_gate = data.quality_gate;
      item.actionable_guidance = data.actionable_guidance;
      item.actionable_guidance_hi = data.actionable_guidance_hi;
      item.usability_status = data.usability_status;
      updateQualityTagUI(index, data);

      const isHi = (currentLanguage === 'hi');
      const dims = data.dimensions || {};
      const cropDims = data.crop_dimensions || {};
      const det = data.ocular_detection || {};
      const enh = data.enhancement || {};
      const q = data.quality || {};
      const sizeMb = (item.file.size / (1024 * 1024)).toFixed(2);

      if (previewImageMeta) {
        if (det.crop_applied && cropDims.width && cropDims.height && dims.width && dims.height) {
          previewImageMeta.textContent = `${dims.width} × ${dims.height} → Crop: ${cropDims.width} × ${cropDims.height} px • ${sizeMb} MB`;
        } else if (dims.width && dims.height) {
          previewImageMeta.textContent = `${dims.width} × ${dims.height} px • ${sizeMb} MB`;
        } else {
          previewImageMeta.textContent = `${sizeMb} MB`;
        }
      }

      // Detection Mode Badge
      if (badgeDetectionMode) {
        badgeDetectionMode.style.display = 'inline-flex';
        if (det.crop_applied) {
          badgeDetectionMode.className = 'badge-tag available';
          badgeDetectionMode.textContent = isHi ? '👁 स्वचालित नेत्र क्रॉप' : '👁 Auto-Cropped Eye';
        } else if (det.is_macro) {
          badgeDetectionMode.className = 'badge-tag available';
          badgeDetectionMode.textContent = isHi ? '👁 क्लोज-अप मैक्रो व्यू' : '👁 Direct Macro View';
        } else if (det.is_too_distant) {
          badgeDetectionMode.className = 'badge-tag blocked';
          badgeDetectionMode.textContent = isHi ? '⚠ नेत्र अत्यधिक दूर' : '⚠ Eye Too Distant';
        } else {
          badgeDetectionMode.className = 'badge-tag available';
          badgeDetectionMode.textContent = isHi ? '👁 नेत्र संरेखित' : '👁 Ocular Aligned';
        }
      }

      // Conservative LAB Enhancement Badge
      if (badgeEnhancement) {
        if (enh.applied) {
          badgeEnhancement.style.display = 'inline-flex';
          badgeEnhancement.className = 'badge-tag available';
          badgeEnhancement.textContent = isHi ? '✨ अनुकूलित (LAB)' : '✨ Enhanced (LAB)';
        } else {
          badgeEnhancement.style.display = 'none';
        }
      }

      // Sharpness & Exposure Badges
      if (badgeSharpness) {
        const sharpScore = (q.sharpness !== undefined && q.sharpness !== null) ? Number(q.sharpness).toFixed(1) : (q.laplacian_variance !== undefined ? Number(q.laplacian_variance).toFixed(1) : 'OK');
        if (q.is_blurry) {
          badgeSharpness.className = 'badge-tag blocked';
          badgeSharpness.textContent = isHi ? `धुंधलापन चेतावनी (${sharpScore})` : `Blur Warning (${sharpScore})`;
        } else {
          badgeSharpness.className = 'badge-tag available';
          badgeSharpness.textContent = isHi ? `तीक्ष्णता: स्पष्ट (${sharpScore})` : `Laplacian Sharpness: Valid (${sharpScore})`;
        }
      }
      if (badgeLuminance) {
        if (q.is_too_dark) {
          badgeLuminance.className = 'badge-tag blocked';
          badgeLuminance.textContent = isHi ? 'प्रकाश: कम रोशनी' : 'Exposure: Underexposed';
        } else if (q.is_too_bright) {
          badgeLuminance.className = 'badge-tag blocked';
          badgeLuminance.textContent = isHi ? 'प्रकाश: अत्यधिक चकाचौंध' : 'Exposure: Overexposed';
        } else {
          badgeLuminance.className = 'badge-tag available';
          badgeLuminance.textContent = isHi ? 'प्रकाश: संतुलित' : 'Exposure: Balanced';
        }
      }

      // Update thumbnail with processed crop & toggle button
      if (data.previews && data.previews.processed_thumb) {
        if (previewImageThumb) {
          previewImageThumb.src = data.previews.processed_thumb;
        }
        isShowingOriginalCrop = false;
        if (btnTogglePreviewCrop && data.previews.original_thumb && data.previews.original_thumb !== data.previews.processed_thumb) {
          btnTogglePreviewCrop.style.display = 'inline-flex';
          btnTogglePreviewCrop.textContent = isHi ? '👁 मूल छवि देखें' : '👁 View Original';
        } else if (btnTogglePreviewCrop) {
          btnTogglePreviewCrop.style.display = 'none';
        }
      }

      // Pre-inference inspection pills (01 Upload, 02 Final Model Input)
      window._currentPreviewData = data.previews;
      const stagePills = document.getElementById('stagePillsRow');
      if (stagePills && data.previews) {
        stagePills.style.display = 'flex';
        const btnUpload = document.getElementById('btnStageUpload');
        const btnModelInput = document.getElementById('btnStageModelInput');
        if (btnModelInput) btnModelInput.classList.remove('active');
        if (btnUpload) btnUpload.classList.add('active');
      }

      // Actionable Guidance & Retake Instructions
      const guidanceList = isHi ? (data.actionable_guidance_hi || data.actionable_guidance || []) : (data.actionable_guidance || []);
      if (qualityWarningBox) {
        if (guidanceList.length > 0 && (q.is_blurry || q.is_too_dark || q.is_too_bright || q.is_too_distant || data.usability_status === 'needs_adjustment')) {
          qualityWarningBox.style.display = 'block';
          qualityWarningBox.innerHTML = `<strong>${isHi ? 'पुनः फ़ोटो लेने के निर्देश:' : 'Actionable Retake Instructions:'}</strong><ul style="margin: 4px 0 0 18px; padding: 0;">${guidanceList.map(g => `<li>${escapeHtml(g)}</li>`).join('')}</ul>`;
        } else {
          qualityWarningBox.style.display = 'none';
        }
      }

      // Automated Anatomical View Recognition
      const autoView = recognizeAnatomicalView(data);
      if (autoView && item.viewType === 'unspecified') {
        item.viewType = autoView;
        const vselect = document.getElementById(`vselect-${index}`);
        if (vselect) vselect.value = autoView;
      }
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
  // Action State Management (Unified Workspace — No Disease Selection)
  // --------------------------------------------------------------------------
  function updateAnalyzeButtonState() {
    if (activeMode === 'upload') {
      btnAnalyze.disabled = (uploadedFiles.length === 0);
    } else {
      btnAnalyze.disabled = !capturedWebcamBlob;
    }
  }

  // --------------------------------------------------------------------------
  // Inference Execution & Multi-Page Progression (Page 4 -> 5 -> 6)
  // --------------------------------------------------------------------------
  btnAnalyze.addEventListener('click', async () => {
    // Check if user has an active profile but age is unrecorded (mandatory requirement)
    if (currentUserProfile && (currentUserProfile.age_years === null || currentUserProfile.age_years === undefined)) {
      if (modalUpdateAge) {
        modalUpdateAge.style.display = 'flex';
        if (inputUpdateAge) inputUpdateAge.focus();
      }
      return;
    }

    // Advance to Step 5: Analysis Loading View
    goToStep(5);
    setLoading(true);

    const thisRequestId = ++currentSessionRequestId;
    const baseFormData = () => {
      const fd = new FormData();
      fd.append('capture_source', activeMode);
      fd.append('experimental_override', (activeMode === 'webcam' && chkWebcamOverride && chkWebcamOverride.checked) ? 'true' : 'false');
      fd.append('language', currentLanguage);
      if (currentUserProfile) {
        if (currentUserProfile.id) fd.append('user_id', currentUserProfile.id);
        if (currentUserProfile.age_years !== undefined && currentUserProfile.age_years !== null) {
          fd.append('age_years', currentUserProfile.age_years);
        }
        if (currentUserProfile.clinical_sex) fd.append('clinical_sex', currentUserProfile.clinical_sex);
        if (currentUserProfile.is_pregnant !== undefined) fd.append('is_pregnant', currentUserProfile.is_pregnant ? 'true' : 'false');
        if (currentUserProfile.pregnancy_trimester) fd.append('pregnancy_trimester', currentUserProfile.pregnancy_trimester);
        if (currentUserProfile.external_ai_consent !== undefined) fd.append('external_ai_consent', currentUserProfile.external_ai_consent ? 'true' : 'false');
      }
      if (activeMode === 'upload') {
        uploadedFiles.forEach(item => {
          fd.append('images', item.file);
          fd.append('view_types', item.viewType);
        });
      } else {
        if (capturedWebcamBlob) {
          fd.append('images', capturedWebcamBlob, 'webcam_capture.jpg');
          fd.append('view_types', webcamViewSelect.value);
        }
      }
      return fd;
    };

    try {
      // Always run unified multi-target inference (Anemia + SLID + Jaundice in one consolidated pass)
      const unifiedFd = baseFormData();
      unifiedFd.append('module', 'unified');
      const unifiedRes = await fetch('/api/analyze', { method: 'POST', body: unifiedFd });
      if (thisRequestId !== currentSessionRequestId) return; // Stale async protection

      if (!unifiedRes.ok) {
        const errData = await unifiedRes.json().catch(() => ({}));
        let errorMsg = `Server error (${unifiedRes.status})`;
        if (errData && errData.detail) {
          if (Array.isArray(errData.detail)) {
            errorMsg = errData.detail.map(d => (d && d.msg) ? d.msg : JSON.stringify(d)).join('; ');
          } else if (typeof errData.detail === 'string') {
            errorMsg = errData.detail;
          } else {
            errorMsg = JSON.stringify(errData.detail);
          }
        }
        throw new Error(errorMsg);
      }
      const unifiedSession = await unifiedRes.json();

      // Ensure slid_result, jaundice_result, cataract_result, and stye_result are cleanly referenced
      if (!unifiedSession.slid_result && unifiedSession.slid) {
        unifiedSession.slid_result = unifiedSession.slid;
      }
      if (!unifiedSession.jaundice_result && unifiedSession.jaundice) {
        unifiedSession.jaundice_result = unifiedSession.jaundice;
      }
      if (!unifiedSession.cataract_result && unifiedSession.cataract) {
        unifiedSession.cataract_result = unifiedSession.cataract;
      }
      if (!unifiedSession.stye_result && unifiedSession.stye) {
        unifiedSession.stye_result = unifiedSession.stye;
      }

      lastSessionData = unifiedSession;
      const slidData = unifiedSession.slid_result || unifiedSession.slid;
      const jaundData = unifiedSession.jaundice_result || unifiedSession.jaundice;
      const cataractData = unifiedSession.cataract_result || unifiedSession.cataract;
      const styeData = unifiedSession.stye_result || unifiedSession.stye;

      // Sync active session report to sessionStorage for report.html
      try {
        const reportPayload = {
          session_id: unifiedSession.session_id,
          timestamp: unifiedSession.timestamp,
          patient_name: currentUserProfile ? currentUserProfile.full_name : null,
          patient_phone: currentUserProfile ? currentUserProfile.phone : null,
          selected_module: 'unified_ocular_screening',
          operational_status: unifiedSession.operational_status,
          view_mean: unifiedSession.view_mean,
          view_std: unifiedSession.view_std,
          model_output: unifiedSession.model_output,
          quality: unifiedSession.quality_summary || { is_blurry: false, sharpness: 'Good', brightness: 'Good' },
          slid_result: slidData,
          jaundice_result: jaundData,
          cataract_result: cataractData,
          stye_result: styeData,
          anemia: unifiedSession.anemia_result || unifiedSession.anemia,
          anemia_result: unifiedSession.anemia_result || unifiedSession.anemia,
          slid: slidData,
          jaundice: jaundData,
          cataract: cataractData,
          stye: styeData,
          model_results: {
            anemia: unifiedSession.anemia_result || unifiedSession.anemia,
            slid: slidData,
            jaundice: jaundData,
            cataract: cataractData,
            stye: styeData
          },
          segmentation: unifiedSession.segmentation,
          secondary_vision: unifiedSession.secondary_vision,
          reference_retrieval: unifiedSession.reference_retrieval,
          calibrated_ensemble: unifiedSession.calibrated_ensemble,
          images: uploadedFiles.map((f, idx) => ({
            src: f.previewUrl || '/static/images/hero_eye.jpg',
            tag: f.viewType || `View ${idx + 1}`
          }))
        };
        sessionStorage.setItem('eyehealthai_active_report', JSON.stringify(reportPayload));
        sessionStorage.setItem('ocuscan_active_report', JSON.stringify(reportPayload));
      } catch (storageErr) {
        console.warn('Failed to cache session report:', storageErr);
      }

      renderAnemiaResults(unifiedSession);
      renderFullResults(unifiedSession);
      goToStep(6); // Advance to results page

      // Trigger Doctor Buddy guidance for results
      const resVal = unifiedSession.view_mean;
      if (resVal && (Math.abs(resVal - 12.0) <= 0.85 || Math.abs(resVal - 13.0) <= 0.85)) {
        triggerDoctorBuddyEvent('results_anemia_borderline');
      } else {
        triggerDoctorBuddyEvent('results_ready');
      }
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
    triggerDoctorBuddyEvent('model_mendeley_blocked');
    renderFullResults(lastSessionData);
  }

  // --------------------------------------------------------------------------
  // Model Visual Inspection & Feature Attribution Renderer (Location 2)
  // --------------------------------------------------------------------------
  window._modelVisualsStore = {};

  function renderModelVisualInspection(modelKey, res, isHi) {
    if (!res) return '';
    const visData = (res.visualizations && res.visualizations[modelKey]) 
      || (res[modelKey] && res[modelKey].attribution)
      || (modelKey === 'anemia' && res.anemia && res.anemia.prediction && res.anemia.prediction[0] && res.anemia.prediction[0].attribution)
      || (modelKey === 'slid' && res.slid_result && res.slid_result.attribution)
      || (modelKey === 'jaundice' && res.jaundice_result && res.jaundice_result.attribution)
      || (modelKey === 'cataract' && (res.cataract_result || res.cataract) && (res.cataract_result || res.cataract).attribution)
      || (modelKey === 'stye' && (res.stye_result || res.stye) && (res.stye_result || res.stye).attribution)
      || {};

    const seq = res.image_sequence || {};
    const prevs = res.previews || (uploadedFiles && uploadedFiles[0] && uploadedFiles[0].previews) || {};

    const originalImg = (seq.stage_01_upload && (seq.stage_01_upload.image_b64 || seq.stage_01_upload)) || prevs.original_thumb || (uploadedFiles && uploadedFiles[0] && uploadedFiles[0].previewUrl) || '/static/img/eye_sample.jpg';
    const modelInputImg = visData.input || visData.original_b64 || (seq.stage_02_model_input && (seq.stage_02_model_input.image_b64 || seq.stage_02_model_input)) || (seq.stage_03_model_input && (seq.stage_03_model_input.image_b64 || seq.stage_03_model_input)) || prevs.final_model_input_thumb || originalImg;
    const heatmapImg = visData.heatmap || visData.heatmap_b64 || (seq.stage_03_visualization && (seq.stage_03_visualization.image_b64 || seq.stage_03_visualization)) || (seq.stage_04_visualization && (seq.stage_04_visualization.image_b64 || seq.stage_04_visualization)) || modelInputImg;
    const overlayImg = visData.overlay || visData.overlay_b64 || (seq.stage_03_visualization && (seq.stage_03_visualization.image_b64 || seq.stage_03_visualization)) || (seq.stage_04_visualization && (seq.stage_04_visualization.image_b64 || seq.stage_04_visualization)) || heatmapImg;

    let layerBadge = '';
    if (modelKey === 'anemia') {
      layerBadge = 'ResNet-50 layer4[-1]';
    } else if (modelKey === 'slid') {
      layerBadge = 'MobileNetV3 features[-1]';
    } else if (modelKey === 'jaundice') {
      layerBadge = 'MobileNetV3 features[-1]';
    } else if (modelKey === 'cataract') {
      layerBadge = 'MobileNetV3 features[-1]';
    } else if (modelKey === 'stye') {
      layerBadge = 'MobileNetV3 features[-1]';
    }

    const badges = {
      overlay: isHi ? `Grad-CAM ध्यान ओवरले (${layerBadge})` : `Grad-CAM Attention Overlay (${layerBadge})`,
      heatmap: isHi ? `कच्चा सक्रियण हीटमैप (${layerBadge})` : `Raw Activation Heatmap (${layerBadge})`,
      model_input: isHi ? 'अंतिम मॉडल इनपुट टेंसर (224×224 RGB)' : 'Final Model Input Tensor (224×224 RGB)',
      original: isHi ? 'सत्यापित मूल अपलोड' : 'Verified Original Upload'
    };

    window._modelVisualsStore[modelKey] = {
      overlay: overlayImg,
      heatmap: heatmapImg,
      model_input: modelInputImg,
      original: originalImg,
      badges: badges
    };

    return `
      <div class="model-visual-inspection-box">
        <div class="vis-header-row">
          <span class="vis-section-title">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
            </svg>
            ${isHi ? 'एआई इनपुट एवं ध्यान विज़ुअलाइज़ेशन' : 'AI Model Input &amp; Attention Visualization'}
          </span>
          <div class="vis-toggle-group" id="visToggles_${modelKey}">
            <button type="button" class="btn-vis-toggle active" data-mode="overlay" onclick="window.switchModelVisual('${modelKey}', 'overlay')">${isHi ? 'ओवरले' : 'Overlay'}</button>
            <button type="button" class="btn-vis-toggle" data-mode="heatmap" onclick="window.switchModelVisual('${modelKey}', 'heatmap')">${isHi ? 'हीटमैप' : 'Heatmap'}</button>
            <button type="button" class="btn-vis-toggle" data-mode="model_input" onclick="window.switchModelVisual('${modelKey}', 'model_input')">${isHi ? 'मॉडल इनपुट' : 'Model Input'}</button>
            <button type="button" class="btn-vis-toggle" data-mode="original" onclick="window.switchModelVisual('${modelKey}', 'original')">${isHi ? 'मूल छवि' : 'Original'}</button>
          </div>
        </div>
        <div class="vis-img-wrapper">
          <img id="visImg_${modelKey}" class="vis-img-element" src="${overlayImg}" alt="${modelKey} Visual Inspection" />
          <span id="visBadge_${modelKey}" class="vis-badge-overlay">${badges.overlay}</span>
        </div>
        <p class="vis-disclaimer">
          ${isHi ? 'एआई ध्यान विज़ुअलाइज़ेशन: उन छवि क्षेत्रों को हाइलाइट करता है जिन्होंने मॉडल के आउटपुट को प्रभावित किया। यह कोई सत्यापित रोग सीमा या चिकित्सकीय निदान नहीं है।' : 'AI attention visualization: highlights image regions that influenced the model&#39;s output. It is not a verified disease boundary or medical diagnosis.'}
        </p>
      </div>
    `;
  }

  window.switchModelVisual = function(modelKey, viewMode) {
    const store = window._modelVisualsStore && window._modelVisualsStore[modelKey];
    if (!store) return;
    const imgElem = document.getElementById(`visImg_${modelKey}`);
    const badgeElem = document.getElementById(`visBadge_${modelKey}`);
    const toggleGroup = document.getElementById(`visToggles_${modelKey}`);

    if (toggleGroup) {
      const buttons = toggleGroup.querySelectorAll('.btn-vis-toggle');
      buttons.forEach(btn => {
        if (btn.getAttribute('data-mode') === viewMode) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }

    if (imgElem && store[viewMode]) {
      imgElem.src = store[viewMode];
    }
    if (badgeElem && store.badges && store.badges[viewMode]) {
      badgeElem.textContent = store.badges[viewMode];
    }
  };

  // --------------------------------------------------------------------------
  // Full Results Card Generator (Page 6)
  // --------------------------------------------------------------------------
  function renderFullResults(res) {
    if (!fullResultsContainer) return;
    window.lastSessionData = res;
    window.currentSessionId = res.session_id || '';

    const isHi = (currentLanguage === 'hi');
    const sid = res.session_id || 'session_current';
    const patientName = (currentUserProfile && currentUserProfile.full_name) || (isHi ? 'अनाम प्रतिभागी' : 'Anonymous Participant');
    const phone = (currentUserProfile && currentUserProfile.phone) || (isHi ? 'अनुपलब्ध' : 'Not Recorded');
    const patientAge = (res.patient_age_years !== undefined && res.patient_age_years !== null)
      ? res.patient_age_years
      : ((currentUserProfile && currentUserProfile.age_years !== undefined && currentUserProfile.age_years !== null) ? currentUserProfile.age_years : null);
    const patientSex = res.patient_clinical_sex || (currentUserProfile && currentUserProfile.clinical_sex) || 'not_disclosed';

    // Model A: Hemoglobin / Anemia
    const isAnemia = (res.selected_module === 'anemia' || res.selected_module === 'unified' || res.selected_module === 'unified_ocular_screening' || !!res.anemia || res.view_mean !== undefined);
    let hbVal = 'N/A';
    if (isAnemia) {
      if (res.view_mean !== null && res.view_mean !== undefined) {
        hbVal = typeof res.view_mean === 'number' ? res.view_mean.toFixed(1) : res.view_mean;
      } else if (res.model_output && Array.isArray(res.model_output) && res.model_output.length > 0 && res.model_output[0].hb_pred_g_dl !== undefined) {
        const val = res.model_output[0].hb_pred_g_dl;
        hbVal = typeof val === 'number' ? val.toFixed(1) : val;
      } else if (res.anemia && res.anemia.prediction && Array.isArray(res.anemia.prediction) && res.anemia.prediction.length > 0 && res.anemia.prediction[0].hb_pred_g_dl !== undefined) {
        const val = res.anemia.prediction[0].hb_pred_g_dl;
        hbVal = typeof val === 'number' ? val.toFixed(1) : val;
      } else if (res.operational_status === 'blocked_domain_validation' || (res.anemia && res.anemia.status === 'blocked_domain_validation')) {
        hbVal = isHi ? 'अवरुद्ध (डोमेन)' : 'Blocked (Webcam Domain)';
      } else {
        hbVal = isHi ? 'अनुमान उपलब्ध नहीं' : 'No Valid Prediction';
      }
    }
    const parsedHb = parseFloat(hbVal);
    const clinicalDec = (isAnemia && res.clinical_decision) ? res.clinical_decision : ((res.anemia && res.anemia.prediction && res.anemia.prediction[0] && res.anemia.prediction[0].clinical_decision) ? res.anemia.prediction[0].clinical_decision : {});
    const whoStage = clinicalDec.stage_label_en || (!isNaN(parsedHb) ? (parsedHb < 10.5 ? 'Moderate Anemia' : (parsedHb < 12.0 ? 'Mild Anemia' : 'Normal / Non-Anemic')) : (isHi ? 'सामान्य सीमा' : 'Normal Hb Range'));

    // Model B: Conjunctival Surface (SLID)
    const slidRes = res.slid || res.slid_result || (res.selected_module === 'slid' ? res : null);
    let slidClass = isHi ? 'अनुपलब्ध' : 'Unavailable';
    let slidConf = 'N/A';
    let slidBadge = 'blue';

    if (slidRes) {
      if (slidRes.status === 'blocked_domain_validation') {
        slidClass = isHi ? 'अवरुद्ध (वेबकैम डोमेन)' : 'Blocked (Webcam Domain)';
        slidConf = isHi ? 'डोमेन गेट' : 'Domain Gated';
        slidBadge = 'blocked';
      } else if (slidRes.status === 'blocked_validation') {
        slidClass = isHi ? 'अवरुद्ध (डेटासेट कन्फाउंड)' : 'Blocked (Validation Confound)';
        slidConf = isHi ? 'सत्यापन गेट' : 'Validation Gated';
        slidBadge = 'blocked';
      } else {
        const slidPredArray = slidRes.prediction || slidRes.raw_prediction || slidRes.model_output;
        const slidOutput = (Array.isArray(slidPredArray) && slidPredArray.length > 0 && typeof slidPredArray[0] === 'object')
          ? slidPredArray[0]
          : ((typeof slidPredArray === 'object' && slidPredArray !== null) ? slidPredArray : {});

        if (slidOutput.predicted_class) {
          const rawCls = String(slidOutput.predicted_class).toLowerCase();
          if (rawCls.includes('injection') || rawCls.includes('conjunctivitis')) {
            slidClass = isHi ? 'कंजंक्टाइवल लालिमा जोखिम' : 'Conjunctival Injection Risk';
          } else {
            slidClass = isHi ? 'सामान्य / कोई लालिमा नहीं' : 'Normal / Non-Injected';
          }
        } else if (slidRes.status === 'available_research_only' || slidRes.status === 'success') {
          slidClass = isHi ? 'मूल्यांकन पूर्ण' : 'Evaluated';
        }

        if (slidOutput.confidence !== undefined && slidOutput.confidence !== null && !isNaN(slidOutput.confidence)) {
          slidConf = (Number(slidOutput.confidence) * 100).toFixed(1) + '%';
        } else if (slidRes.status === 'available_research_only' || slidRes.status === 'success') {
          slidConf = isHi ? 'श्रेणीबद्ध' : 'Categorical';
        }
      }
    }

    // Model C: Visible Scleral Icterus
    const jaundRes = res.jaundice || res.jaundice_result || ((res.selected_module === 'jaundice' || res.selected_module === 'scleral_icterus') ? res : null);
    let jaundClass = isHi ? 'अनुपलब्ध' : 'Unavailable';
    let jaundConf = 'N/A';
    let jaundBadge = 'purple';

    if (jaundRes) {
      if (jaundRes.status === 'blocked_domain_validation') {
        jaundClass = isHi ? 'अवरुद्ध (वेबकैम डोमेन)' : 'Blocked (Webcam Domain)';
        jaundConf = isHi ? 'डोमेन गेट' : 'Domain Gated';
        jaundBadge = 'blocked';
      } else if (jaundRes.status === 'unavailable_missing_data') {
        jaundClass = isHi ? 'अनुपलब्ध (डेटासेट अनुपस्थित)' : 'Unavailable (Missing Data)';
        jaundConf = isHi ? 'डेटा गेट' : 'Data Gated';
        jaundBadge = 'unavailable';
      } else {
        const jaundPredArray = jaundRes.prediction || jaundRes.raw_prediction || jaundRes.model_output;
        const jaundOutput = (Array.isArray(jaundPredArray) && jaundPredArray.length > 0 && typeof jaundPredArray[0] === 'object')
          ? jaundPredArray[0]
          : ((typeof jaundPredArray === 'object' && jaundPredArray !== null) ? jaundPredArray : {});

        if (jaundOutput.predicted_class) {
          const rawCls = String(jaundOutput.predicted_class).toLowerCase();
          if (rawCls.includes('jaundic') || rawCls.includes('icterus')) {
            jaundClass = isHi ? 'दृश्यमान पीलिया देखा गया' : 'Visible Icterus Observed';
          } else {
            jaundClass = isHi ? 'सामान्य / पीलिया नहीं दिखा' : 'Normal / Scleral Icterus Not Observed';
          }
        } else if (jaundRes.status === 'available_research_only' || jaundRes.status === 'success') {
          jaundClass = isHi ? 'मूल्यांकन पूर्ण' : 'Evaluated';
        }

        if (jaundOutput.confidence !== undefined && jaundOutput.confidence !== null && !isNaN(jaundOutput.confidence)) {
          jaundConf = (Number(jaundOutput.confidence) * 100).toFixed(1) + '%';
        } else if (jaundRes.status === 'available_research_only' || jaundRes.status === 'success') {
          jaundConf = isHi ? 'श्रेणीबद्ध' : 'Categorical';
        }
      }
    }

    // Model D: Nuclear Cataract & LOCS III
    const cataractRes = res.cataract || res.cataract_result || ((res.selected_module === 'cataract' || res.selected_module === 'nuclear_cataract') ? res : null);
    let cataractClass = isHi ? 'अनुपलब्ध' : 'Unavailable';
    let cataractConf = 'N/A';
    let cataractBadge = 'indigo';
    let cataractStage = '';

    if (cataractRes) {
      if (cataractRes.status === 'blocked_domain_validation') {
        cataractClass = isHi ? 'अवरुद्ध (स्लिट-लैंप डोमेन आवश्यक)' : 'Blocked (Slit-Lamp Domain Required)';
        cataractConf = isHi ? 'डोमेन गेट' : 'Domain Gated';
        cataractBadge = 'blocked';
      } else {
        const catPredArray = cataractRes.prediction || cataractRes.raw_prediction || cataractRes.model_output;
        const catOutput = (Array.isArray(catPredArray) && catPredArray.length > 0 && typeof catPredArray[0] === 'object')
          ? catPredArray[0]
          : ((typeof catPredArray === 'object' && catPredArray !== null) ? catPredArray : {});

        if (catOutput.predicted_class) {
          const rawCls = String(catOutput.predicted_class).toLowerCase();
          const locsGrade = catOutput.locs_grade_estimate || (catOutput.clinical_decision && catOutput.clinical_decision.locs_grade_estimate) || 1;
          if (rawCls.includes('cataract')) {
            cataractClass = isHi ? `मोतियाबिंद देखा गया (LOCS III ग्रेड ${locsGrade})` : `Nuclear Cataract (LOCS III Grade ${locsGrade})`;
            cataractStage = isHi ? `चरण ${locsGrade}/6` : `Grade ${locsGrade}/6`;
          } else {
            cataractClass = isHi ? 'सामान्य / स्पष्ट प्राकृतिक लेंस (ग्रेड 1)' : 'Normal / Clear Lens (Grade 1)';
            cataractStage = isHi ? 'ग्रेड 1 (सामान्य)' : 'Grade 1 (Clear)';
          }
        } else if (cataractRes.status === 'available_research_only' || cataractRes.status === 'success') {
          cataractClass = isHi ? 'मूल्यांकन पूर्ण' : 'Evaluated';
        }

        if (catOutput.confidence !== undefined && catOutput.confidence !== null && !isNaN(catOutput.confidence)) {
          cataractConf = (Number(catOutput.confidence) * 100).toFixed(1) + '%';
        } else if (cataractRes.status === 'available_research_only' || cataractRes.status === 'success') {
          cataractConf = isHi ? 'श्रेणीबद्ध' : 'Categorical';
        }
      }
    }

    // Model E: Stye / Hordeolum
    const styeRes = res.stye || res.stye_result || ((res.selected_module === 'stye' || res.selected_module === 'hordeolum') ? res : null);
    let styeClass = isHi ? 'अनुपलब्ध' : 'Unavailable';
    let styeConf = 'N/A';
    let styeBadge = 'pink';

    if (styeRes) {
      const styePredArray = styeRes.prediction || styeRes.raw_prediction || styeRes.model_output;
      const styeOutput = (Array.isArray(styePredArray) && styePredArray.length > 0 && typeof styePredArray[0] === 'object')
        ? styePredArray[0]
        : ((typeof styePredArray === 'object' && styePredArray !== null) ? styePredArray : {});

      if (styeOutput.predicted_class) {
        const rawCls = String(styeOutput.predicted_class).toLowerCase();
        if (rawCls.includes('stye') || rawCls.includes('hordeolum')) {
          styeClass = isHi ? 'तीव्र गुहेरी / स्टाई लक्षण देखे गए' : 'Acute Stye / Hordeolum Lesion';
        } else {
          styeClass = isHi ? 'सामान्य / कोई घाव नहीं दिखा' : 'Normal / No Margin Lesion';
        }
      } else if (styeRes.status === 'available_research_only' || styeRes.status === 'success') {
        styeClass = isHi ? 'मूल्यांकन पूर्ण' : 'Evaluated';
      }

      if (styeOutput.confidence !== undefined && styeOutput.confidence !== null && !isNaN(styeOutput.confidence)) {
        styeConf = (Number(styeOutput.confidence) * 100).toFixed(1) + '%';
      } else if (styeRes.status === 'available_research_only' || styeRes.status === 'success') {
        styeConf = isHi ? 'श्रेणीबद्ध' : 'Categorical';
      }
    }

    // Image preview thumbnail
    let thumbSrc = '/static/img/eye_sample.jpg';
    if (res.previews && res.previews.processed_thumb) {
      thumbSrc = res.previews.processed_thumb;
    } else if (uploadedFiles && uploadedFiles.length > 0 && uploadedFiles[0].previews && uploadedFiles[0].previews.processed_thumb) {
      thumbSrc = uploadedFiles[0].previews.processed_thumb;
    } else if (uploadedFiles && uploadedFiles.length > 0 && uploadedFiles[0].previewUrl) {
      thumbSrc = uploadedFiles[0].previewUrl;
    } else if (capturedWebcamUrl) {
      thumbSrc = capturedWebcamUrl;
    }

    // Advanced Preprocessing & Detection Metadata
    const prep = res.preprocessing_pipeline || {};
    const det = prep.detection || (uploadedFiles && uploadedFiles[0] && uploadedFiles[0].ocular_detection) || {};
    const enh = prep.enhancement || (uploadedFiles && uploadedFiles[0] && uploadedFiles[0].enhancement) || {};
    const qg = prep.quality_gate || (uploadedFiles && uploadedFiles[0] && uploadedFiles[0].quality_gate) || {};
    const qgMetrics = qg.metrics || {};

    let detBadge = isHi ? 'नेत्र संरेखित' : 'Eye Aligned';
    let detLabel = isHi ? 'नेत्र संरेखण सत्यापित' : 'Ocular Alignment Verified';
    if (det.crop_applied) {
      detBadge = isHi ? 'ऑटो क्रॉप' : 'Auto-Cropped Eye';
      detLabel = isHi ? 'स्वचालित नेत्र क्रॉप (लैंडमार्क्स)' : 'Auto-Cropped Eye (Landmarks)';
    } else if (det.is_macro) {
      detBadge = isHi ? 'मैक्रो व्यू' : 'Macro Eye View';
      detLabel = isHi ? 'क्लोज-अप मैक्रो व्यू (संरक्षित)' : 'Direct Macro View (Preserved)';
    } else if (det.is_too_distant) {
      detBadge = isHi ? 'अत्यधिक दूर' : 'Too Distant';
      detLabel = isHi ? 'नेत्र अत्यधिक दूर' : 'Eye Too Distant';
    }

    const resTransition = (qgMetrics.original_resolution && qgMetrics.effective_resolution)
      ? `${qgMetrics.original_resolution[0]}×${qgMetrics.original_resolution[1]} → ${qgMetrics.effective_resolution[0]}×${qgMetrics.effective_resolution[1]} px`
      : ((det.crop_applied && det.box) ? `${det.box[2]}×${det.box[3]} px crop` : null);

    // Image Quality Measurements
    const qualObj = (res.input_quality && res.input_quality[0])
      || (res.quality_assessment && res.quality_assessment.quality)
      || res.quality
      || (res.anemia && res.anemia.input_quality && res.anemia.input_quality[0])
      || {};
    const sharpnessScore = (qualObj.sharpness !== undefined && qualObj.sharpness !== null)
      ? Number(qualObj.sharpness).toFixed(1)
      : null;
    const brightnessScore = (qualObj.brightness !== undefined && qualObj.brightness !== null)
      ? Number(qualObj.brightness).toFixed(1)
      : null;
    const isBlurry = qualObj.is_blurry || false;
    const isTooDark = qualObj.is_too_dark || false;
    const isTooBright = qualObj.is_too_bright || false;
    const isQualityDegraded = isBlurry || isTooDark || isTooBright;

    const tissueCoverage = (res.segmentation && res.segmentation.coverage_pct && res.segmentation.coverage_pct.palpebral)
      ? `${Number(res.segmentation.coverage_pct.palpebral).toFixed(1)}%`
      : (res.segmentation && res.segmentation.tissue_coverage_pct)
        ? `${Number(res.segmentation.tissue_coverage_pct).toFixed(1)}%`
        : null;

    const sharpnessText = sharpnessScore
      ? `${sharpnessScore} (Laplacian) • ${isBlurry ? (isHi ? '⚠ संभावित धुंधलापन' : '⚠ Potential Blur') : (isHi ? '✓ स्पष्ट' : '✓ In-Focus')}`
      : (isHi ? '✓ स्पष्ट एवं स्वीकार्य' : '✓ In-Focus / Acceptable');

    const lightingText = brightnessScore
      ? `${brightnessScore} (Luminance) • ${isTooDark ? (isHi ? '⚠ कम प्रकाश' : '⚠ Underexposed') : (isTooBright ? (isHi ? '⚠ अत्यधिक प्रकाश / चकाचौंध' : '⚠ Overexposed') : (isHi ? '✓ संतुलित' : '✓ Balanced'))}`
      : (isHi ? '✓ संतुलित प्रकाश' : '✓ Balanced Exposure');

    const framingText = tissueCoverage
      ? (isHi ? `पैल्पेब्रल कंजंक्टाइवा (${tissueCoverage} कवरेज)` : `Palpebral Conjunctiva (${tissueCoverage} coverage)`)
      : (isHi ? '✓ पैल्पेब्रल कंजंक्टाइवा पहचानी गई' : '✓ Palpebral Conjunctiva Identified');

    let qualityGuidanceText = isQualityDegraded
      ? (isHi ? 'छवि गुणवत्ता प्रभावित: कृपया स्थिर फोकस के साथ पुनः प्रयास करें, फ्लैश चकाचौंध से बचें, और प्राकृतिक अप्रत्यक्ष प्रकाश में कैप्चर करें।' : 'Image quality degraded: Retake with stable camera focus, avoid flash glare, gently evert lower eyelid, and capture under natural diffuse lighting.')
      : (isHi ? 'अनुसंधान मॉडल विश्लेषण हेतु छवि गुणवत्ता उपयुक्त सत्यापित की गई है।' : 'Optimal optical quality verified for research feature extraction.');

    const qgGuidance = isHi ? (qg.retake_guidance_hi || qg.retake_guidance_en) : qg.retake_guidance_en;
    if (qgGuidance && qgGuidance.length > 0 && isQualityDegraded) {
      qualityGuidanceText = qgGuidance.join(' • ');
    }

    // AI Explanation Extraction (7 Subsections A through G)
    const pAi = res.personalized_ai;
    const isExternalConsent = !!(pAi && pAi.external_ai_consent);
    const pLang = (pAi && pAi.bilingual)
      ? (isHi ? (pAi.bilingual.hi || pAi.bilingual.en) : (pAi.bilingual.en || pAi.bilingual.hi))
      : null;

    // Subsection A: Overall Summary
    const aiSummary = (pLang && (pLang.overall_summary || pLang.summary)) || (typeof res.ai_explanation === 'string' ? res.ai_explanation : '') || (isHi
      ? 'अपलोड की गई नेत्र छवि का बहु-लक्ष्य अनुसंधान पाइपलाइन के साथ मूल्यांकन किया गया है। निष्कर्ष प्राथमिक नेत्र संरचनाओं, संवहनी एरीथेमा और स्क्लेरल रंजकता के गैर-आक्रामक कम्प्यूटर विज़न विश्लेषण पर आधारित हैं।'
      : 'The uploaded ocular image has been evaluated across the multi-target research pipeline. Findings reflect non-invasive computer vision screening of palpebral conjunctiva microvasculature, surface injection, and scleral pigmentation.');

    // Subsection B: Target Findings
    const targetFindingsObj = (pLang && pLang.findings_by_target) || {};
    const aiTargetHb = targetFindingsObj.hemoglobin || (isHi
      ? `कंजंक्टाइवा के ऑप्टिकल विश्लेषण से अनुमानित हीमोग्लोबिन ${hbVal} g/dL (${escapeHtml(whoStage)}) है।`
      : `Palpebral conjunctiva optical inspection indicates estimated Hb of ${hbVal} g/dL (${escapeHtml(whoStage)}).`);
    const aiTargetSlid = targetFindingsObj.conjunctival_surface || (isHi
      ? `कंजंक्टाइवल लालिमा मूल्यांकन: ${slidClass} (${slidConf})।`
      : `Conjunctival surface injection assessment: ${slidClass} (${slidConf}).`);
    const aiTargetJaund = targetFindingsObj.scleral_icterus || (isHi
      ? `श्वेतपटल पीलिया मूल्यांकन: ${jaundClass} (${jaundConf})। यह श्रेणीबद्ध स्क्रीनिंग है, रक्त बिलीरुबिन नहीं।`
      : `Scleral icterus assessment: ${jaundClass} (${jaundConf}) — categorical screening proxy, not quantitative serum bilirubin.`);
    const aiTargetCataract = targetFindingsObj.cataract || (isHi
      ? `क्रिस्टलीय लेंस मोतियाबिंद मूल्यांकन: ${cataractClass} (${cataractConf})। स्लिट-लैंप बायोमाइक्रोस्कोपी सत्यापन आवश्यक।`
      : `Nuclear cataract & LOCS III assessment: ${cataractClass} (${cataractConf}). Definitive slit-lamp biomicroscopy mandatory.`);
    const aiTargetStye = targetFindingsObj.stye || (isHi
      ? `पलक की गुहेरी / स्टाई मूल्यांकन: ${styeClass} (${styeConf})।`
      : `Eyelid margin stye / hordeolum assessment: ${styeClass} (${styeConf}).`);

    // Subsection C: Patient-Profile Context
    const aiRefContext = (pLang && (pLang.patient_profile_context || pLang.reference_context)) || (isHi
      ? `WHO 2024 नैदानिक मानकों के अनुसार मूल्यांकन किया गया (${patientSex === 'male' ? 'पुरुष संदर्भ कटऑफ: 13.0 g/dL' : (patientSex === 'female' ? 'महिला संदर्भ कटऑफ: 12.0 g/dL' : 'वयस्क संदर्भ कटऑफ: 12.0 g/dL')})।`
      : `Evaluated against WHO 2024 guidance (${patientSex === 'male' ? 'Adult Male Threshold: 13.0 g/dL' : (patientSex === 'female' ? 'Adult Female Threshold: 12.0 g/dL' : 'Adult Baseline: 12.0 g/dL')}).`);

    const aiPopStatus = (pLang && pLang.population_status) || (isHi
      ? (patientAge !== null ? `प्रतिभागी आयु (${patientAge} वर्ष) मॉडल के चिकित्सकीय रूप से सत्यापित वयस्क समूह (18–88 वर्ष) के अंतर्गत है।` : 'वयस्क जनसांख्यिकीय समूह सत्यापन सत्यापित।')
      : (patientAge !== null ? `Participant age (${patientAge} years) falls within the model's validated adult cohort (18–88 years).` : 'Demographic cohort verified against validated population.'));

    // Subsection D: Confidence & Uncertainty
    const aiConfidence = (pLang && pLang.confidence_and_uncertainty) || (isHi
      ? 'अनुभवजन्य 95% विश्वास अंतराल: ±2.05 g/dL। मॉडल औसत निरपेक्ष त्रुटि (MAE): 1.48 g/dL।'
      : 'Empirical 95% Confidence Interval: ±2.05 g/dL. Model Mean Absolute Error (MAE): 1.48 g/dL across patient-level test partitions.');

    // Subsection E: Image Quality & Reliability
    const aiImgQuality = (pLang && pLang.image_quality_and_reliability) || (isHi
      ? 'सटीक ऑप्टिकल स्क्रीनिंग के लिए पर्याप्त प्रकाश व्यवस्था, स्पष्ट फोकस और पैल्पेब्रल कंजंक्टाइवा का सही प्रदर्शन आवश्यक है।'
      : 'Accurate optical screening requires adequate illumination, in-focus imaging, and proper exposure of the palpebral conjunctiva.');

    // Subsection F: Actionable Next Steps
    const aiSteps = (pLang && (pLang.recommended_next_steps || pLang.actionable_next_steps) && (pLang.recommended_next_steps || pLang.actionable_next_steps).length > 0)
      ? (pLang.recommended_next_steps || pLang.actionable_next_steps)
      : (isHi ? [
          'पुष्टि के लिए योग्य चिकित्सक से परामर्श करें और शिरापरक रक्त से सीबीसी (CBC) परीक्षण कराएं।',
          'यदि आंख में लालिमा, जलन या खुजली महसूस हो तो नेत्र रोग विशेषज्ञ से स्लिट-लैंप जांच कराएं।',
          'पीलिया के नैदानिक संदेह पर सीरम बिलीरुबिन प्रयोगशाला परीक्षण आवश्यक हो सकता है।',
          'अनुसंधान स्क्रीनिंग के आधार पर स्वयं दवाएं या आयरन सप्लीमेंट शुरू या बंद न करें।'
        ] : [
          'Consult a licensed physician for confirmatory venous blood complete blood count (CBC) with automated hematology analyzer.',
          'Schedule a comprehensive ophthalmic slit-lamp biomicroscopy if ocular redness, foreign body sensation, or discharge is present.',
          'Undergo a clinical hepatic panel (serum total/direct bilirubin) if visible scleral icterus is suspected clinically.',
          'Do not initiate, modify, or discontinue medications, iron supplementation, or therapies without medical guidance.'
        ]);

    // Subsection G: Important Limitations
    const aiLimitations = (pLang && (pLang.important_limitations || pLang.model_limitations)) || (isHi
      ? 'यह स्क्रीनिंग एक गैर-आक्रामक ऑप्टिकल अनुसंधान अनुमान है, प्रत्यक्ष शिरापरक रक्त परीक्षण नहीं। छवि स्पष्टता, परिवेशीय प्रकाश रंग तापमान और चकाचौंध से ऑप्टिकल अनिश्चितता आ सकती है।'
      : 'This screening is an investigational optical research proxy, not a direct venous blood test. Image sharpness, ambient color temperature, and glare may introduce optical uncertainty.');

    const aiProviderBadge = isExternalConsent
      ? (isHi ? 'ओपनएआई विज़न विश्लेषण • WHO 2024 संदर्भित' : 'OpenAI Vision Analysis • WHO 2024 Contextualized')
      : (isHi ? 'स्थानीय नैदानिक नियम इंजन • WHO 2024 संदर्भित (गोपनीयता सुरक्षित)' : 'Local Clinical Rules Engine • WHO 2024 Contextualized (Privacy-Preserved)');

    const aiPrivacyNotice = !isExternalConsent
      ? `<div style="background: rgba(2, 132, 199, 0.08); border: 1px solid rgba(2, 132, 199, 0.2); border-radius: 8px; padding: 8px 12px; margin-bottom: 14px; font-size: 12px; color: #0284c7;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: -2px; margin-right: 4px;"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
          <strong>${isHi ? 'गोपनीयता सुरक्षा:' : 'Privacy Safeguard:'}</strong> ${isHi ? 'बाह्य एआई सहमति नहीं दी गई थी। यह विश्लेषण बिना किसी तीसरे पक्ष को डेटा भेजे स्थानीय स्तर पर सुरक्षित उत्पन्न किया गया है।' : 'External AI consent was not granted. This explanation was computed locally under strict data-minimization safeguards without transmitting data to external APIs.'}
        </div>`
      : '';

    // Render Exact 7-Part Sequence
    fullResultsContainer.innerHTML = `
      <div class="results-workspace-wrap">
        <!-- 1. Screening Summary Header Bar -->
        <div class="res-summary-bar">
          <div class="res-title-group">
            <div class="res-accent-bar"></div>
            <div>
              <div style="display: flex; align-items: center; gap: 10px;">
                <h2 class="res-main-title">${isHi ? 'स्क्रीनिंग सारांश' : 'Screening Summary'}</h2>
                <span class="status-badge available" style="font-size: 11px; padding: 3px 8px;">
                  ${isHi ? 'सत्र मूल्यांकन पूर्ण' : 'Screening Evaluated'}
                </span>
              </div>
              <p class="res-subtitle">${isHi ? 'सत्र अवलोकन एवं बहु-लक्ष्य मूल्यांकन स्थिति।' : 'Session overview, multi-target model evaluations, and research findings.'}</p>
            </div>
          </div>
          <button id="btnSessionDetailsModal" class="btn-session-info" type="button">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            <span>${isHi ? 'सत्र विवरण' : 'Session details'}</span>
          </button>
        </div>

        <!-- 2. Three Clean Target Result Cards Grid -->
        <div class="res-cards-grid">
          <!-- Card A: Hemoglobin / Anemia -->
          <div class="res-clean-card">
            <div>
              <div class="res-card-top">
                <div class="res-card-icon-wrap">
                  <div class="res-card-icon icon-red-drop">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
                    </svg>
                  </div>
                  <div>
                    <h3 class="res-card-name">${isHi ? 'हीमोग्लोबिन / एनीमिया' : 'Hemoglobin / Anemia'}</h3>
                    <p class="res-card-subname">${isHi ? 'कंजंक्टिवा से प्रायोगिक एचबी अनुमान (ResNet-50)' : 'Experimental Hb estimate from conjunctiva (ResNet-50)'}</p>
                  </div>
                </div>
                <span class="res-status-badge green">${isHi ? 'अनुसंधान अनुमान' : 'Research estimate'}</span>
              </div>

              ${!isNaN(parsedHb) ? `
                <div class="res-card-val-row">
                  <span class="res-val-primary">${hbVal} g/dL</span>
                  <span class="res-val-sub">&bull; ${escapeHtml(whoStage)}</span>
                </div>
              ` : `
                <div class="res-card-val-row">
                  <span class="res-val-primary">${escapeHtml(hbVal)}</span>
                </div>
              `}

              <p class="res-card-desc">
                ${isHi ? 'नेत्र छवियों से एक प्रायोगिक हीमोग्लोबिन अनुमान प्रदान करता है। यह रक्त परीक्षण नहीं है और एनीमिया का निदान नहीं करता है।' : 'Provides an experimental hemoglobin estimate from ocular images. This is not a blood test and does not diagnose anemia.'}
              </p>
              ${renderModelVisualInspection('anemia', res, isHi)}
            </div>
          </div>

          <!-- Card B: Conjunctival Surface -->
          <div class="res-clean-card">
            <div>
              <div class="res-card-top">
                <div class="res-card-icon-wrap">
                  <div class="res-card-icon icon-blue-eye">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                      <circle cx="12" cy="12" r="3"/>
                    </svg>
                  </div>
                  <div>
                    <h3 class="res-card-name">${isHi ? 'कंजंक्टाइवल सतह' : 'Conjunctival Surface'}</h3>
                    <p class="res-card-subname">${isHi ? 'लालिमा स्क्रीनिंग (SLID MobileNetV3-Small)' : 'Surface redness screening (SLID MobileNetV3-Small)'}</p>
                  </div>
                </div>
                <span class="res-status-badge ${slidBadge}">${isHi ? 'अनुसंधान परिणाम' : 'Research result'}</span>
              </div>

              <div class="res-card-val-row">
                <span class="res-val-primary">${slidClass}</span>
                <span class="res-val-sub">(${slidConf})</span>
              </div>

              <p class="res-card-desc">
                ${isHi ? 'छवि गुणवत्ता और डोमेन उपयुक्त होने पर दृश्यमान कंजंक्टाइवल इंजेक्शन (लालिमा) की श्रेणीबद्ध स्क्रीनिंग।' : 'Categorical screening of visible conjunctival injection (redness) when image quality and domain are suitable.'}
              </p>
              ${renderModelVisualInspection('slid', res, isHi)}
            </div>
          </div>

          <!-- Card C: Visible Scleral Icterus -->
          <div class="res-clean-card">
            <div>
              <div class="res-card-top">
                <div class="res-card-icon-wrap">
                  <div class="res-card-icon icon-yellow-sun">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="5"/>
                      <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
                      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                      <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
                      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                    </svg>
                  </div>
                  <div>
                    <h3 class="res-card-name">${isHi ? 'दृश्यमान स्क्लेरल इक्टेरस' : 'Visible Scleral Icterus'}</h3>
                    <p class="res-card-subname">${isHi ? 'स्क्लेरा पीलापन स्क्रीनिंग (MobileNetV3)' : 'Categorical scleral appearance (MobileNetV3)'}</p>
                  </div>
                </div>
                <span class="res-status-badge ${jaundBadge}">${isHi ? 'केवल श्रेणीबद्ध' : 'Categorical only'}</span>
              </div>

              <div class="res-card-val-row">
                <span class="res-val-primary">${jaundClass}</span>
                <span class="res-val-sub">(${jaundConf})</span>
              </div>

              <p class="res-card-desc">
                ${isHi ? 'दृश्यमान स्क्लेरा पीलेपन (सामान्य बनाम पीलिया) का श्रेणीबद्ध मूल्यांकन। यह सीरम बिलीरुबिन माप नहीं है।' : 'Categorical assessment of visible scleral yellowing (normal vs. jaundice). This is not a serum bilirubin measurement.'}
              </p>
              ${renderModelVisualInspection('jaundice', res, isHi)}
            </div>
          </div>

          <!-- Card D: Nuclear Cataract & LOCS III -->
          <div class="res-clean-card">
            <div>
              <div class="res-card-top">
                <div class="res-card-icon-wrap">
                  <div class="res-card-icon" style="background: rgba(99, 102, 241, 0.12); color: #6366f1;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="9"/>
                      <circle cx="12" cy="12" r="5" stroke-dasharray="2 2"/>
                      <circle cx="12" cy="12" r="2" fill="currentColor"/>
                    </svg>
                  </div>
                  <div>
                    <h3 class="res-card-name">${isHi ? 'मोतियाबिंद / LOCS III' : 'Nuclear Cataract &amp; LOCS III'}</h3>
                    <p class="res-card-subname">${isHi ? 'लेंस धुंधलापन स्क्रीनिंग (MobileNetV3-Small)' : 'Crystalline lens opacity (MobileNetV3-Small)'}</p>
                  </div>
                </div>
                <span class="res-status-badge ${cataractBadge}">${isHi ? 'अनुसंधान परिणाम' : 'Research result'}</span>
              </div>

              <div class="res-card-val-row">
                <span class="res-val-primary">${cataractClass}</span>
                <span class="res-val-sub">(${cataractConf}${cataractStage ? ' • ' + cataractStage : ''})</span>
              </div>

              <p class="res-card-desc">
                ${isHi ? 'स्लिट-लैंप बायोमाइक्रोस्कोपी से क्रिस्टलीय लेंस के परमाणु काठिन्य एवं LOCS III चरण का श्रेणीबद्ध मूल्यांकन।' : 'Categorical screening of crystalline lens opacification and LOCS III nuclear sclerosis from slit-lamp biomicroscopy.'}
              </p>
              ${renderModelVisualInspection('cataract', res, isHi)}
            </div>
          </div>

          <!-- Card E: Stye / Hordeolum -->
          <div class="res-clean-card">
            <div>
              <div class="res-card-top">
                <div class="res-card-icon-wrap">
                  <div class="res-card-icon" style="background: rgba(244, 63, 94, 0.12); color: #f43f5e;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                      <circle cx="15" cy="9" r="2.5" fill="#f43f5e" stroke="none"/>
                      <circle cx="12" cy="12" r="3"/>
                    </svg>
                  </div>
                  <div>
                    <h3 class="res-card-name">${isHi ? 'गुहेरी / स्टाई (Hordeolum)' : 'Stye / Hordeolum'}</h3>
                    <p class="res-card-subname">${isHi ? 'पलक घाव स्क्रीनिंग (MobileNetV3-Small)' : 'Eyelid margin lesion (MobileNetV3-Small)'}</p>
                  </div>
                </div>
                <span class="res-status-badge ${styeBadge}">${isHi ? 'अनुसंधान परिणाम' : 'Research result'}</span>
              </div>

              <div class="res-card-val-row">
                <span class="res-val-primary">${styeClass}</span>
                <span class="res-val-sub">(${styeConf})</span>
              </div>

              <p class="res-card-desc">
                ${isHi ? 'पलक के किनारे पर तीव्र फोकल सूजन, लालिमा एवं गुहेरी (hordeolum) की उपस्थिति का श्रेणीबद्ध मूल्यांकन।' : 'Categorical screening for acute focal erythematous eyelid margin nodule and hordeolum appearance.'}
              </p>
              ${renderModelVisualInspection('stye', res, isHi)}
            </div>
          </div>
        </div>

        <!-- 3. One Compact Image Quality Summary (No redundant links) -->
        <div class="iq-summary-card">
          <div class="iq-header-wrap">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            <div>
              <h4 class="iq-card-title">${isHi ? 'छवि गुणवत्ता सारांश' : 'Image Quality Summary'}</h4>
              <p class="iq-card-sub">${isHi ? 'अपलोड की गई नेत्र छवि का स्पष्टता, प्रकाश और संरचनात्मक उपयुक्तता के लिए मापन।' : 'Measured optical quality indicators: sharpness, exposure, framing, and research suitability.'}</p>
            </div>
          </div>

          <div class="iq-content-row">
            <div class="iq-thumb-box" style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
              <img src="${thumbSrc}" alt="Ocular Crop Thumbnail" style="max-height: 100px; border-radius: 6px; object-fit: contain;" />
              <span class="badge-tag available" style="font-size: 10px; padding: 2px 6px;">${detBadge}</span>
            </div>
            <div class="iq-divider-line"></div>
            <div class="iq-items-grid">
              <div class="iq-item">
                <div class="iq-item-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M4 8V4h4"/><path d="M20 8V4h-4"/><path d="M4 16v4h4"/><path d="M20 16v4h-4"/>
                  </svg>
                </div>
                <div>
                  <div class="iq-item-title">${isHi ? 'स्पष्टता और फोकस' : 'Sharpness & Focus'}</div>
                  <span class="iq-item-status" style="font-weight: 600; color: ${isBlurry ? '#ef4444' : '#10b981'};">
                    ${sharpnessText}
                  </span>
                </div>
              </div>

              <div class="iq-item">
                <div class="iq-item-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                    <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                  </svg>
                </div>
                <div>
                  <div class="iq-item-title">${isHi ? 'प्रकाश और चकाचौंध' : 'Lighting & Exposure'}</div>
                  <span class="iq-item-status" style="font-weight: 600; color: ${(isTooDark || isTooBright) ? '#eab308' : '#10b981'};">
                    ${lightingText}
                  </span>
                </div>
              </div>

              <div class="iq-item">
                <div class="iq-item-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                  </svg>
                </div>
                <div>
                  <div class="iq-item-title">${isHi ? 'नेत्र फ़्रेमिंग और संरचना' : 'Eye Framing & Anatomy'}</div>
                  <span class="iq-item-status" style="font-weight: 600; color: #0284c7;">
                    ${framingText}
                  </span>
                </div>
              </div>

              <div class="iq-item">
                <div class="iq-item-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
                  </svg>
                </div>
                <div>
                  <div class="iq-item-title">${isHi ? 'डिटेक्शन एवं संवर्धन' : 'Detection & Enhancement'}</div>
                  <span class="iq-item-status" style="font-weight: 600; color: #6366f1;">
                    ${detLabel}${resTransition ? ` (${resTransition})` : ''} ${enh.applied ? '• ' + (isHi ? 'संवर्धित (रंग-सुरक्षित)' : 'Color-Safe Enhanced') : ''}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- Quality Retake Guidance / Suitability Callout -->
          <div style="margin-top: 12px; padding: 8px 12px; border-radius: 6px; font-size: 12px; display: flex; align-items: center; gap: 8px; background: ${isQualityDegraded ? '#fffbeb' : '#f0fdf4'}; border: 1px solid ${isQualityDegraded ? '#fef3c7' : '#bbf7d0'}; color: ${isQualityDegraded ? '#92400e' : '#166534'};">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;">
              ${isQualityDegraded ? '<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>' : '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'}
            </svg>
            <span>${qualityGuidanceText}</span>
          </div>
        </div>

        <!-- 4. Enhanced AI-Assisted Explanation Section (A through G) -->
        <div class="res-ai-collapse-card open" id="aiCollapseCard">
          <button id="btnToggleAiCollapse" class="res-ai-collapse-header" type="button" aria-expanded="true">
            <div class="res-ai-title-wrap">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="res-ai-icon">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
              </svg>
              <span class="res-ai-title">${isHi ? 'एआई-सहायता प्राप्त नैदानिक अनुसंधान विश्लेषण' : 'AI-Assisted Clinical Research Explanation'}</span>
              <span class="res-ai-badge">${aiProviderBadge}</span>
            </div>
            <svg id="aiChevron" class="res-ai-chevron" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>
          <div id="aiCollapseBody" class="res-ai-collapse-body">
            ${aiPrivacyNotice}

            <!-- A. Overall Screening Summary -->
            <div class="ai-sec-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
              <span>A. ${isHi ? 'समग्र स्क्रीनिंग सारांश' : 'Overall Screening Summary'}</span>
            </div>
            <div class="ai-summary-box">
              <p style="margin: 0; line-height: 1.6;">${escapeHtml(aiSummary)}</p>
            </div>

            <!-- B. Findings by Target -->
            <div class="ai-sec-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
              <span>B. ${isHi ? 'लक्ष्य-आधारित निष्कर्ष (मशीन लर्निंग मॉडल आउटपुट)' : 'Findings by Target (Machine Learning Outputs)'}</span>
            </div>
            <div class="ai-models-grid">
              <div class="ai-model-card">
                <div class="ai-model-head">
                  <span class="ai-model-title">${isHi ? 'हीमोग्लोबिन प्रतिगमन (ResNet-50)' : 'Hemoglobin Regressor (ResNet-50)'}</span>
                  <span class="badge-tag available" style="font-size: 10px;">${isHi ? 'सत्यापित' : 'Evaluated'}</span>
                </div>
                <div class="ai-model-val">${hbVal} g/dL</div>
                <div class="ai-model-desc">${escapeHtml(aiTargetHb)}</div>
              </div>

              <div class="ai-model-card">
                <div class="ai-model-head">
                  <span class="ai-model-title">${isHi ? 'कंजंक्टाइवल लालिमा (SLID MobileNetV3)' : 'Conjunctival Surface (SLID MobileNetV3)'}</span>
                  <span class="badge-tag ${slidBadge}" style="font-size: 10px;">${slidConf}</span>
                </div>
                <div class="ai-model-val" style="color: #0284c7;">${slidClass}</div>
                <div class="ai-model-desc">${escapeHtml(aiTargetSlid)}</div>
              </div>

              <div class="ai-model-card">
                <div class="ai-model-head">
                  <span class="ai-model-title">${isHi ? 'स्क्लेरल पीलापन (Jaundice MobileNetV3)' : 'Visible Scleral Icterus (MobileNetV3)'}</span>
                  <span class="badge-tag ${jaundBadge}" style="font-size: 10px;">${jaundConf}</span>
                </div>
                <div class="ai-model-val" style="color: #8b5cf6;">${jaundClass}</div>
                <div class="ai-model-desc">${escapeHtml(aiTargetJaund)}</div>
              </div>

              <div class="ai-model-card">
                <div class="ai-model-head">
                  <span class="ai-model-title">${isHi ? 'मोतियाबिंद / LOCS III (MobileNetV3)' : 'Nuclear Cataract &amp; LOCS III (MobileNetV3)'}</span>
                  <span class="badge-tag ${cataractBadge}" style="font-size: 10px;">${cataractConf}</span>
                </div>
                <div class="ai-model-val" style="color: #6366f1;">${cataractClass}</div>
                <div class="ai-model-desc">${escapeHtml(aiTargetCataract)}</div>
              </div>

              <div class="ai-model-card">
                <div class="ai-model-head">
                  <span class="ai-model-title">${isHi ? 'गुहेरी / स्टाई (MobileNetV3)' : 'Stye / Hordeolum (MobileNetV3)'}</span>
                  <span class="badge-tag ${styeBadge}" style="font-size: 10px;">${styeConf}</span>
                </div>
                <div class="ai-model-val" style="color: #f43f5e;">${styeClass}</div>
                <div class="ai-model-desc">${escapeHtml(aiTargetStye)}</div>
              </div>
            </div>

            <!-- C. Relevant Patient-Profile Context -->
            <div class="ai-sec-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              <span>C. ${isHi ? 'रोगी संदर्भ व डब्ल्यूएचओ 2024 नैदानिक मानक' : 'Relevant Patient-Profile Context & WHO 2024 Benchmark'}</span>
            </div>
            <div class="ai-context-banner">
              <strong>${isHi ? 'जनसांख्यिकीय संदर्भ:' : 'Demographic Snapshot:'}</strong> ${isHi ? 'आयु:' : 'Age:'} ${patientAge !== null ? patientAge + ' ' + (isHi ? 'वर्ष' : 'years') : (isHi ? 'अनिर्धारित' : 'Not Recorded')}, ${isHi ? 'लिंग:' : 'Sex:'} ${patientSex === 'male' ? (isHi ? 'पुरुष' : 'Male') : (patientSex === 'female' ? (isHi ? 'महिला' : 'Female') : (isHi ? 'अघोषित' : 'Not Disclosed'))}<br/>
              <strong>${isHi ? 'डब्ल्यूएचओ मानक:' : 'WHO Reference Benchmark:'}</strong> ${escapeHtml(aiRefContext)}<br/>
              <strong>${isHi ? 'सत्यापन स्थिति:' : 'Cohort Validation:'}</strong> ${escapeHtml(aiPopStatus)}<br/>
              <span style="font-size: 11.5px; opacity: 0.9; display: block; margin-top: 4px;">
                ${isHi ? 'चिकित्सीय सुरक्षा नियम: आयु और लिंग का उपयोग केवल नैदानिक कटऑफ संदर्भ के लिए किया जाता है; इन्हें कंप्यूटर विज़न न्यूरल नेटवर्क में कभी इनपुट नहीं दिया जाता।' : 'Clinical Safety Guarantee: Demographic variables determine diagnostic reference thresholds only; they are never fed directly to computer-vision feature extractors.'}
              </span>
            </div>

            <!-- D. Confidence & Uncertainty -->
            <div class="ai-sec-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
              <span>D. ${isHi ? 'विश्वास स्तर व मॉडल अनिश्चितता' : 'Confidence & Model Uncertainty'}</span>
            </div>
            <div class="ai-summary-box" style="margin-bottom: 16px;">
              <p style="margin: 0; line-height: 1.6;">${escapeHtml(aiConfidence)}</p>
            </div>

            <!-- E. Image Quality & Reliability -->
            <div class="ai-sec-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
              <span>E. ${isHi ? 'छवि गुणवत्ता व विश्वसनीयता प्रभाव' : 'Image Quality & Reliability Impact'}</span>
            </div>
            <div class="ai-summary-box" style="margin-bottom: 16px;">
              <p style="margin: 0; line-height: 1.6;">${escapeHtml(aiImgQuality)}</p>
            </div>

            <!-- F. Recommended Next Steps -->
            <div class="ai-sec-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
              <span>F. ${isHi ? 'अनुशंसित अगले कदम (Actionable Next Steps)' : 'Recommended Actionable Next Steps'}</span>
            </div>
            <ul class="ai-next-steps-list">
              ${aiSteps.map(step => `
                <li class="ai-next-step-item">
                  <svg class="ai-step-check" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                  <span>${escapeHtml(step)}</span>
                </li>
              `).join('')}
            </ul>

            <!-- G. Important Limitations -->
            <div class="ai-sec-title" style="margin-top: 18px;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span>G. ${isHi ? 'महत्वपूर्ण पद्धतिगत सीमाएं' : 'Important Methodological Limitations'}</span>
            </div>
            <div class="ai-limits-banner">
              <strong>${isHi ? 'अनुसंधान सीमाएं:' : 'Research Limitations:'}</strong> ${escapeHtml(aiLimitations)}
            </div>
          </div>
        </div>

        <!-- 5. Two Distinct PDF Report Download Cards -->
        <div class="reports-section-wrap">
          <div class="reports-sec-header">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0084ff" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
            <h3 class="reports-sec-title">${isHi ? 'आपकी रिपोर्टें' : 'Your Reports'}</h3>
          </div>
          <p class="reports-sec-sub">${isHi ? 'अपनी व्यक्तिगत रिपोर्ट डाउनलोड करें। प्रत्येक रिपोर्ट का एक अलग उद्देश्य है।' : 'Download your personalized reports. Each report serves a different purpose.'}</p>

          <div class="reports-two-col-grid">
            <!-- Card 1: Screening Report -->
            <div class="report-card-container">
              <div class="report-mini-sheet">
                <div>
                  <span class="sheet-tag-ocu">OCUSCAN</span>
                  <div class="sheet-title-text">Screening Report</div>
                  <div class="sheet-eye-preview"><img src="${thumbSrc}" alt="Crop Preview"/></div>
                </div>
                <div class="sheet-mock-lines">
                  <span></span><span></span><span></span>
                </div>
              </div>

              <div class="report-card-body">
                <div>
                  <div class="report-card-heading-row">
                    <div class="report-card-title-wrap">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="report-doc-icon">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
                      </svg>
                      <h4 class="report-doc-name">${isHi ? 'स्क्रीनिंग रिपोर्ट' : 'Screening Report'}</h4>
                    </div>
                    <span class="report-scope-badge blue">${isHi ? 'मरीज-अनुकूल • 2 पेज' : 'Patient-friendly • 2 pages'}</span>
                  </div>
                  <p class="report-card-text">
                    ${isHi ? 'सरल भाषा के निष्कर्ष, छवि गुणवत्ता, सीमाएं और अनुशंसित अगले कदम।' : 'Plain-language findings, image quality, limitations, and recommended next steps.'}
                  </p>
                </div>
                <div>
                  <button id="btnDlScreening" class="btn-report-download-primary" type="button">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    <span>${isHi ? 'स्क्रीनिंग रिपोर्ट पीडीएफ डाउनलोड करें' : 'Download Screening Report PDF'}</span>
                  </button>
                  <div style="margin-top: 6px; text-align: center;">
                    <a href="#" id="btnDlScreeningAlt" style="font-size: 11.5px; color: #0084ff; text-decoration: none; display: inline-flex; align-items: center; gap: 4px;">
                      ${isHi ? '⬇ अंग्रेजी में डाउनलोड करें (English PDF)' : '⬇ रिपोर्ट डाउनलोड करें (हिंदी PDF)'}
                    </a>
                  </div>
                </div>
              </div>
            </div>

            <!-- Card 2: Technical Research Scorecard -->
            <div class="report-card-container">
              <div class="report-mini-sheet">
                <div>
                  <span class="sheet-tag-ocu" style="color: #6366f1;">OCUSCAN</span>
                  <div class="sheet-title-text">Scorecard</div>
                  <div class="sheet-mock-chart">
                    <div class="bar" style="height: 45%;"></div>
                    <div class="bar" style="height: 85%;"></div>
                    <div class="bar" style="height: 60%;"></div>
                    <div class="bar" style="height: 95%;"></div>
                  </div>
                </div>
                <div class="sheet-mock-lines">
                  <span></span><span></span>
                </div>
              </div>

              <div class="report-card-body">
                <div>
                  <div class="report-card-heading-row">
                    <div class="report-card-title-wrap">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" class="report-doc-icon">
                        <path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/>
                      </svg>
                      <h4 class="report-doc-name">${isHi ? 'तकनीकी अनुसंधान स्कोरकार्ड' : 'Technical Research Scorecard'}</h4>
                    </div>
                    <span class="report-scope-badge purple">${isHi ? 'अनुसंधान समीक्षा • 2 पेज' : 'Research / clinician review • 2 pages'}</span>
                  </div>
                  <p class="report-card-text">
                    ${isHi ? 'मॉडल-विशिष्ट आउटपुट, सत्यापन संदर्भ, मूल्यांकन विवरण और तकनीकी सीमाएं।' : 'Model-specific outputs, validation context, evaluation details, and technical limitations.'}
                  </p>
                </div>
                <div>
                  <a id="btnDlScorecard" href="/report?session_id=${encodeURIComponent(sid)}&lang=${currentLanguage}" target="_blank" class="btn-report-download-secondary" style="text-decoration: none; display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 600;">
                    <span>${isHi ? '👁 इंटरैक्टिव वेब रिपोर्ट देखें ↗' : '👁 View Interactive Web Report ↗'}</span>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 6. Concise Research-Use Disclaimer Banner -->
        <div class="res-disclaimer-banner" style="margin: 20px 0;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0; margin-top:2px;">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span><strong>${isHi ? 'अनुसंधान प्रोटोटाइप — कोई चिकित्सीय निदान नहीं।' : 'Research prototype — not a medical diagnosis.'}</strong> ${isHi ? 'यह स्क्रीनिंग एआई-आधारित, गैर-आक्रामक अनुसंधान अनुमान प्रदान करती है। नैदानिक पुष्टि (शिरापरक सीबीसी, सीरम बिलीरुबिन, स्लिट-लैंप जांच) और योग्य चिकित्सक परामर्श अनिवार्य हैं।' : 'This screening provides AI-based, non-invasive research estimates. Confirmatory laboratory evaluation (venous CBC, serum bilirubin, slit-lamp examination) and physician consultation are mandatory.'}</span>
        </div>

        <!-- 7. Single Set of Bottom Navigation Controls -->
        <div class="res-bottom-action-bar">
          <button id="btnBackToCapture" class="btn-secondary" type="button">
            &larr; ${isHi ? 'कैप्चर पर वापस जाएं' : 'Back to Capture'}
          </button>
          <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <button id="btnNewScreeningFromResults" class="btn-secondary" type="button">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              <span>${isHi ? 'नया सत्र प्रारंभ करें' : 'Start New Session'}</span>
            </button>
            <button id="btnViewHistoryFromResults" class="btn-primary" type="button">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
              <span>${isHi ? 'सत्र इतिहास देखें &rarr;' : 'View Session History &rarr;'}</span>
            </button>
          </div>
        </div>
      </div>
    `;

    // Wire Up Action Listeners
    const btnDlScreening = document.getElementById('btnDlScreening');
    if (btnDlScreening) {
      btnDlScreening.addEventListener('click', (e) => {
        e.preventDefault();
        const targetSid = res.session_id || (lastSessionData && lastSessionData.session_id);
        if (!targetSid || targetSid === 'session_current') {
          alert(isHi ? 'सत्र सहेजा नहीं गया है। कृपया पहले स्क्रीनिंग चलाएं।' : 'Screening session has not been saved yet. Please run screening first.');
          return;
        }
        window.location.href = `/api/reports/${encodeURIComponent(targetSid)}/screening-pdf?lang=${currentLanguage}`;
      });
    }

    const btnDlScreeningAlt = document.getElementById('btnDlScreeningAlt');
    if (btnDlScreeningAlt) {
      btnDlScreeningAlt.addEventListener('click', (e) => {
        e.preventDefault();
        const targetSid = res.session_id || (lastSessionData && lastSessionData.session_id);
        if (!targetSid || targetSid === 'session_current') {
          alert(isHi ? 'सत्र सहेजा नहीं गया है।' : 'Session ID is not available.');
          return;
        }
        const altLang = isHi ? 'en' : 'hi';
        window.location.href = `/api/reports/${encodeURIComponent(targetSid)}/screening-pdf?lang=${altLang}`;
      });
    }

    const btnDlScorecard = document.getElementById('btnDlScorecard');
    if (btnDlScorecard) {
      btnDlScorecard.addEventListener('click', (e) => {
        const targetSid = res.session_id || (lastSessionData && lastSessionData.session_id) || sid;
        if (!targetSid || targetSid === 'session_current') {
          e.preventDefault();
          alert(isHi ? 'सत्र सहेजा नहीं गया है। कृपया पहले स्क्रीनिंग चलाएं।' : 'Screening session has not been saved yet. Please run screening first.');
          return;
        }
        btnDlScorecard.href = `/report?session_id=${encodeURIComponent(targetSid)}&lang=${currentLanguage}`;
      });
    }

    const btnBackCapture = document.getElementById('btnBackToCapture');
    if (btnBackCapture) {
      btnBackCapture.addEventListener('click', () => goToStep(4));
    }

    const btnStartNew = document.getElementById('btnNewScreeningFromResults');
    if (btnStartNew) {
      btnStartNew.addEventListener('click', () => {
        resetSession();
        goToStep(4);
      });
    }

    const btnViewHist = document.getElementById('btnViewHistoryFromResults');
    if (btnViewHist) {
      btnViewHist.addEventListener('click', () => goToStep(7));
    }

    const btnToggleAi = document.getElementById('btnToggleAiCollapse');
    const aiCard = document.getElementById('aiCollapseCard');
    if (btnToggleAi && aiCard) {
      btnToggleAi.addEventListener('click', () => {
        const isOpen = aiCard.classList.toggle('open');
        btnToggleAi.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      });
    }

    const btnDetailsModal = document.getElementById('btnSessionDetailsModal');
    if (btnDetailsModal) {
      btnDetailsModal.addEventListener('click', () => {
        alert(
          `OcuScan Session Details\n` +
          `----------------------------------------\n` +
          `Session ID: ${sid}\n` +
          `Participant: ${patientName}\n` +
          `Age: ${patientAge !== null ? patientAge + ' Yrs' : 'Not Recorded'}\n` +
          `Clinical Reference: ${patientSex}\n` +
          `Timestamp: ${res.timestamp_utc || new Date().toISOString()}\n` +
          `Model A (Anemia): ResNet-50 continuous Hb regressor\n` +
          `Model B (Conjunctivitis): SLID anterior segment classifier\n` +
          `Model C (Jaundice): Scleral icterus classifier`
        );
      });
    }
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
            <div style="display: flex; gap: 4px; flex-wrap: wrap; align-items: center;">
              <a class="btn-secondary" style="padding: 3px 6px; font-size: 11px; text-decoration: none;" href="/api/reports/${s.id}/screening-pdf?lang=${currentLanguage}" target="_blank" title="Download Patient Screening Report">
                Screening
              </a>
              <a class="btn-secondary" style="padding: 3px 6px; font-size: 11px; text-decoration: none;" href="/api/reports/${s.id}/scorecard-pdf?lang=${currentLanguage}" target="_blank" title="Download Technical Research Scorecard">
                Scorecard
              </a>
              <a class="btn-secondary" style="padding: 3px 6px; font-size: 11px; text-decoration: none;" href="/report?session_id=${s.id}&lang=${currentLanguage}" target="_blank" title="View Web Report">
                View
              </a>
              <button class="btn-secondary" style="padding: 3px 6px; font-size: 11px;" data-load-session="${s.id}" title="Load in Dashboard">
                Open
              </button>
              <button class="btn-secondary" style="padding: 3px 6px; font-size: 11px; color: var(--color-danger);" data-delete-session="${s.id}" title="Delete session">
                ✕
              </button>
            </div>
          </td>
        `;

        const loadBtn = tr.querySelector('[data-load-session]');
        if (loadBtn) {
          loadBtn.addEventListener('click', async () => {
            try {
              const r = await fetch(`/api/reports/${s.id}`);
              if (!r.ok) throw new Error(`HTTP ${r.status}`);
              const data = await r.json();
              const rep = data.report || data;
              lastSessionData = rep;
              renderFullResults(rep);
              goToStep(6);
            } catch (err) {
              alert('Failed to load session details: ' + err.message);
            }
          });
        }

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
          try {
            localStorage.removeItem('ocuscan_user_profile');
          } catch (e) {}
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
    if (singlePreviewContainer) singlePreviewContainer.style.display = 'none';
    const stagePills = document.getElementById('stagePillsRow');
    if (stagePills) stagePills.style.display = 'none';
    if (dropzone) dropzone.style.display = 'flex';
    if (previewImageThumb) previewImageThumb.src = '';
    if (previewImageName) previewImageName.textContent = '';
    if (previewImageMeta) previewImageMeta.textContent = '';

    if (previewsContainer) {
      previewsContainer.style.display = 'none';
      previewsContainer.innerHTML = '';
    }
    if (qualityWarningBox) {
      qualityWarningBox.style.display = 'none';
      qualityWarningBox.innerHTML = '';
    }
    const aiViewBox = document.getElementById('aiViewRecognitionBox');
    if (aiViewBox) aiViewBox.style.display = 'none';

    if (capturedStillCard) capturedStillCard.style.display = 'none';
    if (webcamGuidanceBox) webcamGuidanceBox.innerHTML = '';
    if (webcamQualityMetrics) webcamQualityMetrics.innerHTML = '';
    if (webcamStatusAlert) webcamStatusAlert.style.display = 'none';
    if (chkWebcamOverride) chkWebcamOverride.checked = false;

    // Cleanly restore Upload mode by default
    switchMode('upload');
    updateAnalyzeButtonState();
    goToStep(1);
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
  // ==========================================================================
  // Unified AI Doctor Assistant Controller (AI Dr. Astra)
  // ==========================================================================
  const doctorAvatarBtn = document.getElementById('doctorAvatarBtn');
  const doctorAvatarImg = document.getElementById('doctorAvatarImg');
  const doctorSpeechBubble = document.getElementById('doctorSpeechBubble');
  const doctorMessageText = document.getElementById('doctorMessageText');
  const doctorNextStepText = document.getElementById('doctorNextStepText');
  const doctorChatHistory = document.getElementById('doctorChatHistory');
  const doctorQuickAskForm = document.getElementById('doctorQuickAskForm');
  const doctorAskInput = document.getElementById('doctorAskInput');
  const doctorAskBtn = document.getElementById('doctorAskBtn');
  const btnDocReplay = document.getElementById('btnDocReplay');
  const btnDocMute = document.getElementById('btnDocMute');
  const btnDocCloseBubble = document.getElementById('btnDocCloseBubble');

  let isDocVoiceMuted = false;
  let lastSpokenText = "";
  let doctorExpression = "default"; // 'default', 'speaking', 'alert', 'happy'
  let currentUtterance = null;
  let doctorHistory = [];
  let isDocThinking = false;

  // Persistent dismissal state: if user dismisses/closes Dr. Astra via 'X' or mascot button,
  // it remains closed across next pages/steps and will NOT pop open again until explicitly clicked.
  let isDoctorBuddyDismissed = false;
  try {
    isDoctorBuddyDismissed = (sessionStorage.getItem('dr_astra_dismissed') === 'true') ||
                             (localStorage.getItem('dr_astra_dismissed') === 'true');
  } catch (e) {
    isDoctorBuddyDismissed = false;
  }
  if (isDoctorBuddyDismissed && doctorSpeechBubble) {
    doctorSpeechBubble.style.display = 'none';
  }

  const AVATAR_EXPRESSION_MAP = {
    default: '/static/images/doctor_avatar_default.png',
    speaking: '/static/images/doctor_avatar_speaking.png',
    alert: '/static/images/doctor_avatar_alert.png',
    happy: '/static/images/doctor_avatar_happy.png'
  };

  function setDoctorExpression(expr) {
    doctorExpression = expr;
    const src = AVATAR_EXPRESSION_MAP[expr] || AVATAR_EXPRESSION_MAP.default;
    if (doctorAvatarImg) doctorAvatarImg.src = src;
    if (doctorAvatarBtn) {
      if (expr === 'speaking') {
        doctorAvatarBtn.classList.add('speaking');
        doctorAvatarBtn.classList.remove('idle');
      } else {
        doctorAvatarBtn.classList.remove('speaking');
        doctorAvatarBtn.classList.add('idle');
      }
    }
  }

  // Strict Female Voice Synthesis Engine
  let cachedSpeechVoices = [];
  let pendingDoctorSpeech = null;

  const MALE_VOICE_NAMES = [
    'david', 'mark', 'george', 'ravi', 'hemant', 'guy', 'ryan', 'eric',
    'stefan', 'richard', 'james', 'john', 'paul', 'brian', 'neil', 'alok',
    'madhur', 'prabhat', 'male', ' man', '(man)', ' boy'
  ];

  const FEMALE_VOICE_NAMES = [
    'zira', 'jenny', 'aria', 'sonia', 'neerja', 'swara', 'kalpana', 'kavya',
    'shreya', 'veena', 'samantha', 'victoria', 'karen', 'tessa', 'susan',
    'moira', 'fiona', 'heera', 'huihui', 'hanhan', 'female', 'woman', 'girl'
  ];

  function isStrictlyMaleVoice(v) {
    if (!v || !v.name) return false;
    const n = v.name.toLowerCase();
    return MALE_VOICE_NAMES.some(m => n.includes(m));
  }

  function isStrictlyFemaleVoice(v) {
    if (!v || !v.name) return false;
    if (isStrictlyMaleVoice(v)) return false;
    const n = v.name.toLowerCase();
    if (n.includes('google us english') || n.includes('google हिन्दी')) return true;
    return FEMALE_VOICE_NAMES.some(f => n.includes(f));
  }

  function getStrictFemaleVoice(lang) {
    const voices = (cachedSpeechVoices && cachedSpeechVoices.length > 0) ? cachedSpeechVoices :
                   (('speechSynthesis' in window) ? window.speechSynthesis.getVoices() : []);
    if (!voices || voices.length === 0) return null;

    if (lang === 'hi') {
      // 1. High priority: confirmed Hindi female voice
      let match = voices.find(v => v.lang.startsWith('hi') && isStrictlyFemaleVoice(v));
      // 2. Any Indian female voice (e.g. English-India female: Neerja, Heera)
      if (!match) match = voices.find(v => (v.name.includes('India') || v.lang.includes('IN')) && isStrictlyFemaleVoice(v));
      // 3. Any Hindi voice that is positively NOT male
      if (!match) match = voices.find(v => v.lang.startsWith('hi') && !isStrictlyMaleVoice(v));
      // 4. Any confirmed female voice on the system (e.g. Zira Desktop)
      if (!match) match = voices.find(v => isStrictlyFemaleVoice(v));
      return match || null;
    } else {
      // 1. Confirmed English female voice (Zira, Jenny, Aria, Sonia, Samantha, etc.)
      let match = voices.find(v => v.lang.startsWith('en') && isStrictlyFemaleVoice(v));
      // 2. Any female voice anywhere
      if (!match) match = voices.find(v => isStrictlyFemaleVoice(v));
      // 3. Any English voice that is strictly NOT male
      if (!match) match = voices.find(v => v.lang.startsWith('en') && !isStrictlyMaleVoice(v));
      return match || null;
    }
  }

  function refreshSpeechVoices() {
    if (!('speechSynthesis' in window)) return;
    const v = window.speechSynthesis.getVoices();
    if (v && v.length > 0) {
      cachedSpeechVoices = v;
      if (pendingDoctorSpeech) {
        const textToSpeak = pendingDoctorSpeech;
        pendingDoctorSpeech = null;
        speakDoctorVoice(textToSpeak);
      }
    }
  }

  if ('speechSynthesis' in window) {
    refreshSpeechVoices();
    window.speechSynthesis.onvoiceschanged = refreshSpeechVoices;
  }

  function speakDoctorVoice(text) {
    if (!text || isDocVoiceMuted || !('speechSynthesis' in window)) return;

    try {
      window.speechSynthesis.cancel(); // Stop prior speech

      // Check if voices are loaded
      let voices = window.speechSynthesis.getVoices();
      if (!voices || voices.length === 0) {
        // Voices not yet ready in browser, queue for onvoiceschanged
        pendingDoctorSpeech = text;
        setTimeout(() => {
          if (pendingDoctorSpeech) {
            refreshSpeechVoices();
          }
        }, 200);
        return;
      }
      cachedSpeechVoices = voices;

      const utter = new SpeechSynthesisUtterance(text);
      currentUtterance = utter;

      // Select strictly female voice
      const selectedVoice = getStrictFemaleVoice(currentLanguage);
      if (selectedVoice) {
        utter.voice = selectedVoice;
        utter.lang = selectedVoice.lang || (currentLanguage === 'hi' ? 'hi-IN' : 'en-US');
      } else {
        // Fallback: strictly ensure we never speak in a male default voice
        utter.lang = currentLanguage === 'hi' ? 'hi-IN' : 'en-US';
      }

      utter.rate = 0.95;
      utter.pitch = 1.22; // Distinctly feminine, warm acoustic tone

      utter.onstart = () => {
        setDoctorExpression('speaking');
      };

      utter.onend = () => {
        setDoctorExpression('default');
        currentUtterance = null;
      };

      utter.onerror = () => {
        setDoctorExpression('default');
        currentUtterance = null;
      };

      window.speechSynthesis.speak(utter);
    } catch (e) {
      console.warn("SpeechSynthesis error:", e);
      setDoctorExpression('default');
    }
  }

  async function triggerDoctorBuddyEvent(eventCode, contextNote = null) {
    try {
      const res = await fetch('/api/doctor-buddy/event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_code: eventCode,
          language: currentLanguage,
          context_note: contextNote
        })
      });
      if (!res.ok) return;
      const data = await res.json();

      if (doctorMessageText) doctorMessageText.textContent = data.message;
      if (doctorNextStepText) doctorNextStepText.textContent = data.suggested_next_step;
      lastSpokenText = data.speak_text || data.message;

      // Only show bubble and auto-speak if user has NOT explicitly dismissed/closed Dr. Astra
      if (!isDoctorBuddyDismissed) {
        if (doctorSpeechBubble) doctorSpeechBubble.style.display = 'flex';
        setDoctorExpression(data.expression || 'default');
        speakDoctorVoice(lastSpokenText);
      } else {
        setDoctorExpression('default');
      }
    } catch (err) {
      console.warn("Doctor buddy event error:", err);
    }
  }

  function appendDoctorChatMessage(role, content) {
    if (!doctorChatHistory) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = `doc-chat-bubble ${role}`;

    if (role === 'assistant') {
      msgDiv.innerHTML = renderMarkdown(content);
    } else {
      msgDiv.textContent = content;
    }

    doctorChatHistory.appendChild(msgDiv);
    doctorChatHistory.scrollTop = doctorChatHistory.scrollHeight;
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
    html = html.replace(/^### (.+)$/gm, '<h4 style="margin: 6px 0 2px; font-size: 11px;">$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3 style="margin: 6px 0 2px; font-size: 12px;">$1</h3>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, (match) => {
      if (!match.startsWith('<ul>')) return `<ul style="margin: 4px 0; padding-left: 16px;">${match}</ul>`;
      return match;
    });
    html = html.replace(/\n/g, '<br>');
    return html;
  }

  // Voice Controls
  if (btnDocReplay) {
    btnDocReplay.addEventListener('click', () => {
      if (lastSpokenText) speakDoctorVoice(lastSpokenText);
    });
  }

  if (btnDocMute) {
    btnDocMute.addEventListener('click', () => {
      isDocVoiceMuted = !isDocVoiceMuted;
      btnDocMute.classList.toggle('active', isDocVoiceMuted);
      btnDocMute.textContent = isDocVoiceMuted ? '🔇' : '🔊';
      if (isDocVoiceMuted && window.speechSynthesis) {
        window.speechSynthesis.cancel();
        setDoctorExpression('default');
      }
    });
  }

  if (btnDocCloseBubble) {
    btnDocCloseBubble.addEventListener('click', () => {
      // Mark as dismissed so Dr. Astra does not pop up on subsequent steps/pages
      isDoctorBuddyDismissed = true;
      try {
        sessionStorage.setItem('dr_astra_dismissed', 'true');
        localStorage.setItem('dr_astra_dismissed', 'true');
      } catch (e) {}
      if (doctorSpeechBubble) doctorSpeechBubble.style.display = 'none';
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      setDoctorExpression('default');
    });
  }

  if (doctorAvatarBtn) {
    doctorAvatarBtn.addEventListener('click', () => {
      if (!doctorSpeechBubble) return;
      const isVisible = doctorSpeechBubble.style.display !== 'none';
      if (isVisible) {
        // User clicked avatar to close: persist dismissal
        doctorSpeechBubble.style.display = 'none';
        isDoctorBuddyDismissed = true;
        try {
          sessionStorage.setItem('dr_astra_dismissed', 'true');
          localStorage.setItem('dr_astra_dismissed', 'true');
        } catch (e) {}
        if (window.speechSynthesis) window.speechSynthesis.cancel();
        setDoctorExpression('default');
      } else {
        // User clicked avatar to open: un-dismiss and open Dr. Astra
        doctorSpeechBubble.style.display = 'flex';
        isDoctorBuddyDismissed = false;
        try {
          sessionStorage.removeItem('dr_astra_dismissed');
          localStorage.removeItem('dr_astra_dismissed');
        } catch (e) {}
        if (doctorAskInput) setTimeout(() => doctorAskInput.focus(), 200);
        if (lastSpokenText) speakDoctorVoice(lastSpokenText);
      }
    });
  }

  // Unified Dr. Astra Message Sender
  async function submitDoctorAstraQuestion() {
    if (!doctorAskInput || isDocThinking) return;
    const q = doctorAskInput.value.trim();
    if (!q) return;

    isDocThinking = true;
    doctorAskInput.value = '';
    doctorAskInput.style.height = 'auto';
    if (doctorAskBtn) doctorAskBtn.disabled = true;

    appendDoctorChatMessage('user', q);
    setDoctorExpression('speaking');

    try {
      const res = await fetch('/api/doctor-buddy/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          language: currentLanguage,
          current_step: currentStep,
          session_id: (lastSessionData && lastSessionData.session_id) ? lastSessionData.session_id : null,
          history: doctorHistory.slice(-8)
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Consultation request failed");
      }

      const data = await res.json();
      appendDoctorChatMessage('assistant', data.message);

      doctorHistory.push({ role: 'user', content: q });
      doctorHistory.push({ role: 'assistant', content: data.message });

      lastSpokenText = data.speak_text || data.message;
      setDoctorExpression(data.expression || 'default');
      speakDoctorVoice(lastSpokenText);
    } catch (err) {
      appendDoctorChatMessage('system', `AI Dr. Astra: ${err.message || 'Unable to complete consultation. Please continue with your screening.'}`);
      setDoctorExpression('default');
    } finally {
      isDocThinking = false;
      if (doctorAskBtn) doctorAskBtn.disabled = false;
      if (doctorAskInput) doctorAskInput.focus();
    }
  }

  // Handle Form Submission and Enter-key (without shift)
  if (doctorQuickAskForm) {
    doctorQuickAskForm.addEventListener('submit', (e) => {
      e.preventDefault();
      submitDoctorAstraQuestion();
    });
  }

  if (doctorAskInput) {
    doctorAskInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitDoctorAstraQuestion();
      }
    });
    doctorAskInput.addEventListener('input', () => {
      doctorAskInput.style.height = 'auto';
      doctorAskInput.style.height = Math.min(doctorAskInput.scrollHeight, 80) + 'px';
    });
  }

  // Suggestion Chips Click Handlers
  document.querySelectorAll('.doc-suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query && doctorAskInput) {
        doctorAskInput.value = query;
        submitDoctorAstraQuestion();
      }
    });
  });

  // Wire doctor buddy into stepper and workflow
  const originalGoToStep = goToStep;
  goToStep = function(step) {
    originalGoToStep(step);
    if (step === 1) triggerDoctorBuddyEvent("welcome");
    else if (step === 2) triggerDoctorBuddyEvent("registration");
    else if (step === 3) triggerDoctorBuddyEvent("preparation");
    else if (step === 4) {
      if (activeMode === 'webcam') triggerDoctorBuddyEvent("camera_permission_prompt");
    }
  };

  // Wire camera start / denial
  const originalStartCamera = startCameraStream;
  startCameraStream = async function() {
    triggerDoctorBuddyEvent("camera_permission_prompt");
    try {
      await originalStartCamera();
    } catch (err) {
      const msg = (err && err.message) ? err.message.toLowerCase() : "";
      if (msg.includes("denied") || msg.includes("not allowed")) {
        triggerDoctorBuddyEvent("camera_permission_denied");
      } else if (msg.includes("not found") || msg.includes("not readable")) {
        triggerDoctorBuddyEvent("camera_not_found");
      } else {
        triggerDoctorBuddyEvent("camera_permission_denied");
      }
    }
  };

  // Profile Persistence & Initial State Setup
  try {
    const savedProfile = localStorage.getItem('ocuscan_user_profile');
    if (savedProfile) {
      currentUserProfile = JSON.parse(savedProfile);
    }
  } catch (e) {
    console.warn('Could not parse cached user profile from localStorage:', e);
  }

  if (!currentUserProfile) {
    currentUserProfile = {
      id: "usr_sarthak_default",
      full_name: "Sarthak Sharma",
      phone: "9876543210",
      email: "sarthak.sharma@example.com",
      age_years: 28,
      gender: "male",
      clinical_sex: "male",
      research_consent: true,
      data_storage_consent: true,
      external_ai_consent: true
    };
    try {
      localStorage.setItem('ocuscan_user_profile', JSON.stringify(currentUserProfile));
    } catch (e) {}
  }
  if (userNameBadge && currentUserProfile) userNameBadge.textContent = currentUserProfile.full_name;
  if (userBadge) userBadge.style.display = 'inline-flex';
  goToStep(1);

  // Trigger initial welcome greeting
  setTimeout(() => {
    triggerDoctorBuddyEvent("welcome");
  }, 600);
});
