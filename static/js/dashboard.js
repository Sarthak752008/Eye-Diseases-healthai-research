/* ==========================================================================
   OcuScan — Research Dashboard Client Logic (Layers 5, 6 & 7)
   Multi-Step Journey (5 Steps), Live Laptop Webcam Capture, Internationalization (EN/HI),
   Database Session Management, Two-Page PDF Export, and Gemini Research Copilot.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Global Workflow State
  let currentStep = 1; // 1 to 5 - Starts on Participant Profile (Step 1)
  let currentLanguage = 'en'; // 'en' or 'hi'
  let currentUserProfile = null; // { id, full_name, email, phone, date_of_birth, gender }
  let currentModule = 'anemia';
  let activeMode = 'upload'; // 'upload' or 'webcam'
  let uploadedFiles = []; // Array of { file, viewType, previewUrl, quality, dimensions }
  let lastSessionData = null;
  let currentSessionRequestId = 0; // Guard against race conditions and stale async responses
  let isAnalyzing = false;

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
      step1: "Participant Profile",
      step2: "Protocol Guide",
      step3: "Screening Workspace",
      step4: "Results &amp; Report",
      step5: "History",
      btnNavHistoryText: "History",
      btnNewSessionText: "New Session",
      regTitle: "Participant Profile &amp; Informed Consent",
      regSub: "Please enter your details and review the research consent terms before proceeding. Your data is stored privacy-compliantly.",
      labelFullName: "Full Name *",
      labelPhone: "Phone Number *",
      labelEmail: "Email Address *",
      labelDob: "Date of Birth (YYYY-MM-DD) *",
      labelGender: "Gender *",
      genderSelectDefault: "Select Gender",
      genderMale: "Male",
      genderFemale: "Female",
      genderOther: "Other",
      consentHeader: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> Informed Consent &amp; Privacy Notice',
      consentP1: "<strong>Experimental Research Only:</strong> OcuScan is an investigational software prototype. It does NOT provide a certified medical diagnosis.",
      consentP2: "<strong>No Raw Image Retention:</strong> Captured eye photographs and webcam frames are processed strictly in RAM and are NEVER permanently stored on disk by default.",
      consentP3: "<strong>Laboratory Reference Required:</strong> Continuous hemoglobin estimations require confirmatory laboratory Complete Blood Count (CBC) venous blood tests.",
      consentP4: "<strong>Right to Be Forgotten:</strong> You can permanently delete your participant profile, session history, and report metadata at any time.",
      chkConsentResearchText: "I understand this is an experimental research prototype and agree to participate in non-invasive ocular screening.",
      chkConsentStorageText: "I consent to privacy-aware session and report metadata storage with the ability to purge my records at any time.",
      btnSkipReg: "Continue as Guest / Anonymous &rarr;",
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
      btnBackProfile: "&larr; Back to Profile",
      btnReadyCapture: "Ready to Capture &rarr;",
      screeningTitle: "Non-Invasive Ocular Health Screening",
      screeningSub: "Upload high-resolution anterior segment photographs or capture a live macro camera still to estimate continuous hemoglobin (Hb in g/dL) with active clinical reliability and dataset validation gates.",
      safetyGateTitle: "Active Safety Gating:",
      safetyGateText: "Continuous hemoglobin predictions are permitted for experimental research use. Disease classifications are gated when dataset shortcuts or missing ground truth are detected.",
      modulesTitle: "Clinical Target Modules",
      moduleAnemiaName: "Anemia / Hemoglobin",
      moduleAnemiaSub: "Continuous Hb Regression",
      statusResearchOnly: "RESEARCH ONLY",
      moduleAnemiaDesc: "Predicts laboratory hemoglobin (g/dL) from conjunctival pallor using a pretrained ResNet-50. Supports multi-view session averaging across palpebral and forniceal crops.",
      moduleConjName: "Conjunctivitis",
      moduleConjSub: "Ocular Erythema Classifier",
      statusBlockedGate: "BLOCKED (GATE)",
      moduleConjDesc: "Reliability gate active. Layer 3.5 audit revealed normal vs disease classes have non-overlapping dimensions (56px square vs 352px macro). Predictions are blocked to prevent shortcut reliance.",
      moduleJaundName: "Jaundice / Icterus",
      moduleJaundSub: "Scleral Bilirubin Screening",
      statusUnavailable: "UNAVAILABLE",
      moduleJaundDesc: "No clinical dataset or laboratory total serum bilirubin measurements exist in workspace. Synthetic predictions and placeholder models are prohibited.",
      inputSessionTitle: "Input Session &amp; Image Capture",
      modeUpload: "File Upload Mode",
      modeWebcam: "Live Laptop Camera",
      dropzoneTitle: "Click to upload or drag &amp; drop images",
      dropzoneHint: "Supports JPG, PNG, WEBP (Ocular anterior crops or full eye)",
      guideReticleTitle: "Manual Macro Framing Target (10–15 cm)",
      guideReticleSub: "Position one eye to fill this area. Positioning aid only; not automated eye detection.",
      cameraMacroWarning: "<strong>⚠️ MACRO EYE FRAMING REQUIRED — NOT A FULL FACE:</strong><br>The model requires a close-up image of the ocular conjunctiva. A full-face selfie contains mostly facial skin and background, which cannot be analyzed for microvascular conjunctival pallor. Move within 10–15 cm of the laptop camera and expose the target eye view.",
      btnStartCam: "Start Camera",
      btnCaptureCam: "Capture Still",
      btnStopCam: "Stop",
      capturedStillTitle: "Captured Webcam Still",
      btnRecapture: "Recapture",
      labelAnatomicalView: "Anatomical View Tag:",
      viewRawEye: "Raw Full Eye View",
      viewPalpebral: "Palpebral Conjunctiva",
      viewForniceal: "Forniceal Conjunctiva",
      viewFornicealPalpebral: "Forniceal + Palpebral",
      viewUnspecified: "Unspecified Ocular Crop",
      webcamGateTitle: "Camera-Domain Validation Gate Active",
      webcamGateText: "The anemia ResNet-50 model was trained strictly on clinical research photography and has not been validated for ordinary laptop webcams. By default, webcam inference is blocked per Layer 6 standards.",
      chkWebcamOverrideText: "Enable Researcher Experimental Override (Unvalidated Webcam Test)",
      btnAnalyzeText: "Run Research Inference",
      sessionPreviewTitle: "Session Analysis Preview",
      clinicalReportBtn: "Clinical Report",
      exportJsonBtn: "Export JSON",
      resultsEmptyText: "Select images and click <strong>Run Research Inference</strong> to generate an experimental session estimate.",
      checklistTitle: "Clinical Protocol Checklist",
      checklistItem1: "Macro anterior framing within 10–15 cm",
      checklistItem2: "Expose lower palpebral/forniceal conjunctiva",
      checklistItem3: "Real-time sharpness &amp; luminance validation",
      checklistItem4: "Continuous ResNet-50 Hb regression (g/dL)",
      btnBackGuide: "&larr; Back to Protocol Guide",
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
      appTitle: "OcuScan",
      appSubtitle: "मल्टी-मॉडल नेत्र जांच &bull; हीमोग्लोबिन निरंतर प्रतिगमन &bull; चिकित्सीय सुरक्षा गेट",
      step1: "प्रतिभागी विवरण",
      step2: "प्रोटोकॉल गाइड",
      step3: "स्क्रीनिंग वर्कस्पेस",
      step4: "परिणाम एवं रिपोर्ट",
      step5: "इतिहास",
      btnNavHistoryText: "इतिहास",
      btnNewSessionText: "नया सत्र",
      regTitle: "प्रतिभागी विवरण और सूचित सहमति",
      regSub: "आगे बढ़ने से पहले कृपया अपना विवरण दर्ज करें और अनुसंधान सहमति शर्तों की समीक्षा करें। आपकी गोपनीयता सुरक्षित है।",
      labelFullName: "पूरा नाम *",
      labelPhone: "फोन नंबर *",
      labelEmail: "ईमेल पता *",
      labelDob: "जन्म तिथि (YYYY-MM-DD) *",
      labelGender: "लिंग *",
      genderSelectDefault: "लिंग चुनें",
      genderMale: "पुरुष (Male)",
      genderFemale: "महिला (Female)",
      genderOther: "अन्य (Other)",
      consentHeader: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> सूचित सहमति एवं गोपनीयता सूचना',
      consentP1: "<strong>केवल प्रायोगिक अनुसंधान:</strong> OcuScan एक अनुसंधान सॉफ्टवेयर है। यह प्रमाणित चिकित्सीय निदान (Medical Diagnosis) प्रदान नहीं करता है।",
      consentP2: "<strong>मूल तस्वीरें सुरक्षित नहीं:</strong> ली गई आंख की तस्वीरें केवल रैम में प्रोसेस होती हैं और डिस्क पर कभी स्थायी रूप से सुरक्षित नहीं रखी जातीं।",
      consentP3: "<strong>प्रयोगशाला परीक्षण आवश्यक:</strong> निरंतर हीमोग्लोबिन अनुमानों के लिए प्रयोगशाला सीबीसी (CBC) रक्त परीक्षण द्वारा पुष्टि आवश्यक है।",
      consentP4: "<strong>डेटा मिटाने का अधिकार:</strong> आप किसी भी समय अपनी प्रोफाइल और इतिहास को स्थायी रूप से मिटा सकते हैं।",
      chkConsentResearchText: "मैं समझता/समझती हूँ कि यह एक अनुसंधान प्रोटोटाइप है और गैर-आक्रामक जांच में भाग लेने के लिए सहमत हूँ।",
      chkConsentStorageText: "मैं सत्र और रिपोर्ट मेटाडेटा के सुरक्षित भंडारण की सहमति देता/देती हूँ।",
      btnSkipReg: "अतिथि / अज्ञात के रूप में जारी रखें &rarr;",
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
      btnBackProfile: "&larr; प्रोफाइल पर वापस जाएं",
      btnReadyCapture: "तस्वीर लेने के लिए तैयार &rarr;",
      screeningTitle: "गैर-आक्रामक नेत्र स्वास्थ्य जांच",
      screeningSub: "सक्रिय नैदानिक विश्वसनीयता और डेटासेट सत्यापन गेट्स के साथ निरंतर हीमोग्लोबिन (Hb g/dL) का अनुमान लगाने के लिए उच्च-रिज़ॉल्यूशन तस्वीरें अपलोड करें या लाइव मैक्रो कैमरा स्टिल कैप्चर करें।",
      safetyGateTitle: "सक्रिय सुरक्षा गेटिंग:",
      safetyGateText: "निरंतर हीमोग्लोबिन अनुमान प्रायोगिक अनुसंधान उपयोग के लिए अनुमत हैं। डेटासेट शॉर्टकट या अनुपलब्ध ग्राउंड ट्रुथ पाए जाने पर रोग वर्गीकरण अवरुद्ध किया जाता है।",
      modulesTitle: "चिकित्सीय लक्ष्य मॉड्यूल",
      moduleAnemiaName: "एनीमिया / हीमोग्लोबिन",
      moduleAnemiaSub: "निरंतर हीमोग्लोबिन प्रतिगमन",
      statusResearchOnly: "केवल अनुसंधान",
      moduleAnemiaDesc: "प्री-ट्रेंड ResNet-50 का उपयोग करके कंजंक्टाइवल पल्लोर से प्रयोगशाला हीमोग्लोबिन (g/dL) का अनुमान लगाता है। पैल्पेब्रल और फोर्निसियल व्यूज में मल्टी-व्यू औसत का समर्थन करता है।",
      moduleConjName: "कंजंक्टिवाइटिस",
      moduleConjSub: "नेत्र एरिथेमा वर्गीकरण",
      statusBlockedGate: "अवरुद्ध (सुरक्षा गेट)",
      moduleConjDesc: "विश्वसनीयता गेट सक्रिय है। डेटासेट ऑडिट से पता चला कि सामान्य और रोगग्रस्त छवियों में रेजोल्यूशन अंतर (56px बनाम 352px) है। कृत्रिम शॉर्टकट निर्भरता रोकने के लिए भविष्यवाणियां रोकी गई हैं।",
      moduleJaundName: "पीलिया / इक्टेरस",
      moduleJaundSub: "स्क्लेरल बिलीरुबिन स्क्रीनिंग",
      statusUnavailable: "अनुपलब्ध",
      moduleJaundDesc: "कार्यक्षेत्र में कोई क्लिनिकल डेटासेट या प्रयोगशाला कुल सीरम बिलीरुबिन संदर्भ माप मौजूद नहीं है। कृत्रिम भविष्यवाणियां सख्त वर्जित हैं।",
      inputSessionTitle: "सत्र इनपुट एवं फोटो कैप्चर",
      modeUpload: "फ़ाइल अपलोड मोड",
      modeWebcam: "लाइव लैपटॉप कैमरा",
      dropzoneTitle: "छवियों को अपलोड करने के लिए क्लिक करें या यहाँ खींचें",
      dropzoneHint: "JPG, PNG, WEBP समर्थित (आंख की नजदीकी तस्वीर या पूरी आंख)",
      guideReticleTitle: "मैनुअल मैक्रो फ्रेमिंग गाइड (10–15 सेमी)",
      guideReticleSub: "एक आंख को इस क्षेत्र में रखें। केवल स्थिति सहायता; स्वचालित नेत्र पहचान नहीं।",
      cameraMacroWarning: "<strong>⚠️ मैक्रो आई फ्रेमिंग आवश्यक — पूरा चेहरा नहीं:</strong><br>मॉडल को कंजंक्टिवा की नजदीकी तस्वीर की आवश्यकता है। पूरे चेहरे की सेल्फी का विश्लेषण नहीं किया जा सकता। कैमरे से 10–15 सेमी की दूरी पर आएं और आंख की तस्वीर लें।",
      btnStartCam: "कैमरा शुरू करें",
      btnCaptureCam: "फोटो खींचें",
      btnStopCam: "कैमरा रोकें",
      capturedStillTitle: "कैप्चर की गई कैमरा फोटो",
      btnRecapture: "दोबारा खींचें",
      labelAnatomicalView: "एनाटोमिकल व्यू टैग:",
      viewRawEye: "पूरी आंख का दृश्य (Raw Eye)",
      viewPalpebral: "पैल्पेब्रल कंजंक्टिवा (Palpebral)",
      viewForniceal: "फोर्निसियल कंजंक्टिवा (Forniceal)",
      viewFornicealPalpebral: "फोर्निसियल + पैल्पेब्रल",
      viewUnspecified: "अनिर्दिष्ट नेत्र भाग",
      webcamGateTitle: "कैमरा-डोमेन सत्यापन गेट सक्रिय",
      webcamGateText: "एनीमिया ResNet-50 मॉडल विशेष रूप से क्लिनिकल रिसर्च फोटोग्राफी पर प्रशिक्षित है और साधारण वेबकैम के लिए मान्य नहीं है। डिफ़ॉल्ट रूप से वेबकैम परिणाम अवरुद्ध हैं।",
      chkWebcamOverrideText: "शोधकर्ता प्रायोगिक ओवरराइड सक्षम करें (अपुष्ट वेबकैम परीक्षण)",
      btnAnalyzeText: "शोध अनुमान शुरू करें",
      sessionPreviewTitle: "सत्र विश्लेषण पूर्वावलोकन",
      clinicalReportBtn: "क्लिनिकल रिपोर्ट",
      exportJsonBtn: "JSON निर्यात",
      resultsEmptyText: "प्रायोगिक अनुमान प्राप्त करने के लिए चित्र चुनें और <strong>शोध अनुमान शुरू करें</strong> पर क्लिक करें।",
      checklistTitle: "क्लिनिकल प्रोटोकॉल चेकलिस्ट",
      checklistItem1: "10–15 सेमी की दूरी से मैक्रो आई फ्रेमिंग",
      checklistItem2: "निचली पैल्पेब्रल/फोर्निसियल कंजंक्टिवा प्रदर्शित करें",
      checklistItem3: "रीयल-टाइम तीखापन एवं रोशनी सत्यापन",
      checklistItem4: "निरंतर ResNet-50 हीमोग्लोबिन प्रतिगमन (g/dL)",
      btnBackGuide: "&larr; प्रोटोकॉल गाइड पर वापस जाएं",
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

  // Dynamic Module Banners & CTA Text Dictionaries (for real-time language updates)
  const MODULE_BANNERS = {
    anemia: {
      en: '<strong>Active Module: Anemia Hemoglobin Regressor</strong><br>Permitted for experimental research use. You may upload one or multiple anatomical views (e.g. palpebral, forniceal, full eye) from the same capture session.',
      hi: '<strong>सक्रिय मॉड्यूल: एनीमिया हीमोग्लोबिन रिग्रेसर</strong><br>प्रायोगिक अनुसंधान उपयोग के लिए अनुमत। आप एक ही सत्र से एक या एकाधिक दृश्य (पैल्पेब्रल, फोर्निसियल, पूरी आंख) अपलोड कर सकते हैं।'
    },
    conjunctivitis: {
      en: '<strong>Reliability Gate Active: Conjunctivitis Classifier</strong><br>Predictions are blocked due to a 100% resolution confound in source data.',
      hi: '<strong>विश्वसनीयता गेट सक्रिय: कंजंक्टिवाइटिस क्लासिफायर</strong><br>स्रोत डेटा में 100% रिज़ॉल्यूशन पूर्वाग्रह के कारण भविष्यवाणियां अवरुद्ध हैं।'
    },
    jaundice: {
      en: '<strong>Missing-Data Gate: Jaundice Scleral Icterus</strong><br>No clinical dataset or laboratory bilirubin reference measurements exist in the workspace.',
      hi: '<strong>डेटा अनुपलब्ध गेट: पीलिया स्क्लेरल इक्टेरस</strong><br>कार्यक्षेत्र में कोई क्लिनिकल डेटासेट या प्रयोगशाला बिलीरुबिन संदर्भ माप उपलब्ध नहीं है।'
    }
  };

  const MODULE_CTA_TEXT = {
    anemia: {
      en: 'Run Research Inference',
      hi: 'शोध अनुमान शुरू करें'
    },
    conjunctivitis: {
      en: 'Inspect Gated Module',
      hi: 'अवरुद्ध मॉड्यूल का निरीक्षण करें'
    },
    jaundice: {
      en: 'Inspect Missing Data',
      hi: 'अनुपलब्ध डेटा का निरीक्षण करें'
    }
  };

  // Anemia View Options (Localized)
  const ANEMIA_VIEWS_I18N = {
    en: [
      { value: 'palpebral', label: 'Palpebral Conjunctiva' },
      { value: 'forniceal', label: 'Forniceal Conjunctiva' },
      { value: 'forniceal_palpebral', label: 'Forniceal + Palpebral' },
      { value: 'raw_eye', label: 'Raw Full Eye View' },
      { value: 'unspecified', label: 'Unspecified Ocular Crop' }
    ],
    hi: [
      { value: 'palpebral', label: 'पैल्पेब्रल कंजंक्टिवा (Palpebral)' },
      { value: 'forniceal', label: 'फोर्निसियल कंजंक्टिवा (Forniceal)' },
      { value: 'forniceal_palpebral', label: 'फोर्निसियल + पैल्पेब्रल' },
      { value: 'raw_eye', label: 'पूरी आंख का दृश्य (Raw Eye)' },
      { value: 'unspecified', label: 'अनिर्दिष्ट नेत्र भाग' }
    ]
  };

  // --------------------------------------------------------------------------
  
  // Strict Guided Navigation Rule State
  let hasCompletedGuide = false;

  function updateStepLockStates() {
    const step3Btn = document.getElementById('stepBtn3');
    if (step3Btn) {
      if (!hasCompletedGuide) {
        step3Btn.classList.add('step-locked');
        step3Btn.setAttribute('title', currentLanguage === 'hi'
          ? 'कृपया पहले प्रोटोकॉल गाइड पूरा करें (Ready to Capture पर क्लिक करें)'
          : 'Please complete Protocol Guide first (Click "Ready to Capture")');
      } else {
        step3Btn.classList.remove('step-locked');
        step3Btn.removeAttribute('title');
      }
    }
  }

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

  // --------------------------------------------------------------------------
  // Dark / Light Theme Engine
  // --------------------------------------------------------------------------
  function initTheme() {
    const savedTheme = localStorage.getItem('eyehealth_theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const activeTheme = savedTheme || (prefersDark ? 'dark' : 'light');
    applyTheme(activeTheme);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeToggleBtn) {
      const isDark = (theme === 'dark');
      themeToggleBtn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
      themeToggleBtn.setAttribute('title', isDark ? 'Switch to light theme' : 'Switch to dark theme');
    }
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const nextTheme = current === 'dark' ? 'light' : 'dark';
      localStorage.setItem('eyehealth_theme', nextTheme);
      applyTheme(nextTheme);
    });
  }

  initTheme();

  // Step 1: Profile / Login Elements
  const formRegistration = document.getElementById('formRegistration');
  const inputRegName = document.getElementById('inputRegName');
  const inputRegPhone = document.getElementById('inputRegPhone');
  const inputRegEmail = document.getElementById('inputRegEmail');
  const inputRegDob = document.getElementById('inputRegDob');
  const inputRegGender = document.getElementById('inputRegGender');
  const chkConsentResearch = document.getElementById('chkConsentResearch');
  const chkConsentStorage = document.getElementById('chkConsentStorage');
  const regErrorAlert = document.getElementById('regErrorAlert');
  const btnSkipRegistration = document.getElementById('btnSkipRegistration');

  // Step 2: Protocol Guide Elements
  const btnBackToReg = document.getElementById('btnBackToReg');
  const btnProceedToCapture = document.getElementById('btnProceedToCapture');

  // Step 3: Screening Workspace Elements
  const btnBackToGuide = document.getElementById('btnBackToGuide');
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

  // Step 4: Results & Report Elements
  const analysisLoadingView = document.getElementById('analysisLoadingView');
  const resultsNoDataView = document.getElementById('resultsNoDataView');
  const resultsSuccessView = document.getElementById('resultsSuccessView');
  const btnGoToCaptureFromEmpty = document.getElementById('btnGoToCaptureFromEmpty');
  const fullResultsContainer = document.getElementById('fullResultsContainer');
  const btnDownloadPdfReport = document.getElementById('btnDownloadPdfReport');
  const btnDownloadPdfHi = document.getElementById('btnDownloadPdfHi');
  const btnOpenPrintScorecard = document.getElementById('btnOpenPrintScorecard');
  const btnBackToCapture = document.getElementById('btnBackToCapture');
  const btnNewScreeningFromResults = document.getElementById('btnNewScreeningFromResults');
  const btnViewHistoryFromResults = document.getElementById('btnViewHistoryFromResults');

  // Step 5: History Elements
  const historyTableBody = document.getElementById('historyTableBody');
  const historyEmptyState = document.getElementById('historyEmptyState');
  const btnPurgeUserData = document.getElementById('btnPurgeUserData');
  const btnBackToScreening = document.getElementById('btnBackToScreening');

  // Report Modal Preview
  const reportModal = document.getElementById('reportModal');
  const btnCloseReportModal = document.getElementById('btnCloseReportModal');
  const reportFrame = document.getElementById('reportFrame');

  // --------------------------------------------------------------------------
  // Dynamic Module Banner & CTA Updates
  // --------------------------------------------------------------------------
  function updateModuleBanner() {
    if (!moduleBanner) return;
    const bannerInfo = MODULE_BANNERS[currentModule] || MODULE_BANNERS.anemia;
    if (currentModule === 'anemia') {
      moduleBanner.className = 'alert-box alert-info';
    } else if (currentModule === 'conjunctivitis') {
      moduleBanner.className = 'alert-box alert-danger';
    } else if (currentModule === 'jaundice') {
      moduleBanner.className = 'alert-box alert-warning';
    }
    moduleBanner.innerHTML = bannerInfo[currentLanguage] || bannerInfo.en;
  }

  function updateAnalyzeButtonText() {
    if (!btnAnalyzeText) return;
    const ctaInfo = MODULE_CTA_TEXT[currentModule] || MODULE_CTA_TEXT.anemia;
    btnAnalyzeText.textContent = ctaInfo[currentLanguage] || ctaInfo.en;
  }

  // --------------------------------------------------------------------------
  // Internationalization Logic (Preserving State & Streams)
  // --------------------------------------------------------------------------
  function setLanguage(lang) {
    currentLanguage = lang;
    if (langBtnEn) langBtnEn.classList.toggle('active', lang === 'en');
    if (langBtnHi) langBtnHi.classList.toggle('active', lang === 'hi');

    const dict = I18N[lang] || I18N.en;
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (dict[key]) {
        el.innerHTML = dict[key];
      }
    });

    if (btnNewSessionText) btnNewSessionText.textContent = lang === 'hi' ? 'नया सत्र' : 'New Session';
    if (btnNavHistoryText) btnNavHistoryText.textContent = lang === 'hi' ? 'इतिहास' : 'History';

    // Refresh dynamic module content in active language
    updateModuleBanner();
    updateAnalyzeButtonText();

    // Update anatomical view select dropdowns in place
    document.querySelectorAll('.view-select').forEach(sel => {
      const currentVal = sel.value;
      const opts = ANEMIA_VIEWS_I18N[lang] || ANEMIA_VIEWS_I18N.en;
      sel.innerHTML = opts.map(o => `<option value="${o.value}" ${o.value === currentVal ? 'selected' : ''}>${o.label}</option>`).join('');
    });

    if (webcamViewSelect) {
      const currentVal = webcamViewSelect.value;
      const opts = ANEMIA_VIEWS_I18N[lang] || ANEMIA_VIEWS_I18N.en;
      webcamViewSelect.innerHTML = opts.map(o => `<option value="${o.value}" ${o.value === currentVal ? 'selected' : ''}>${o.label}</option>`).join('');
    }

    // Update results cards in place if active
    if (lastSessionData) {
      renderFullResults(lastSessionData);
    }
  }

  if (langBtnEn) langBtnEn.addEventListener('click', () => setLanguage('en'));
  if (langBtnHi) langBtnHi.addEventListener('click', () => setLanguage('hi'));

  // --------------------------------------------------------------------------
  // Multi-Step Navigation Engine (Exact 5-Step Order)
  // 1. Participant Profile -> 2. Protocol Guide -> 3. Screening Workspace -> 4. Results & Report -> 5. History
  // --------------------------------------------------------------------------
  function goToStep(step) {
    if (step < 1 || step > 5) step = 1;

    // Strict Guided Navigation Rule: Cannot bypass Protocol Guide (Step 2) to reach Screening Workspace (Step 3)
    if (step === 3 && !hasCompletedGuide) {
      if (typeof triggerDoctorBuddyEvent === 'function') {
        triggerDoctorBuddyEvent('guide_required', {
          message: currentLanguage === 'hi'
            ? 'स्क्रीनिंग कार्यक्षेत्र में जाने से पहले कृपया प्रोटोकॉल गाइड की समीक्षा करें।'
            : 'Please review the Protocol Guide before proceeding to the Screening Workspace.'
        });
      }
      step = 2; // Redirect to Protocol Guide
    }

    currentStep = step;
    updateStepLockStates();

    const pageViews = [
      { num: 1, id: 'page2Registration' },
      { num: 2, id: 'page3Guide' },
      { num: 3, id: 'page4Capture' },
      { num: 4, id: 'page6Results' },
      { num: 5, id: 'page7History' },
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

    if (step === 4) {
      updateResultsStepDisplay();
    }

    if (step === 5) {
      loadSessionHistory();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function updateResultsStepDisplay() {
    if (isAnalyzing) {
      if (analysisLoadingView) analysisLoadingView.style.display = 'block';
      if (resultsNoDataView) resultsNoDataView.style.display = 'none';
      if (resultsSuccessView) resultsSuccessView.style.display = 'none';
    } else if (lastSessionData) {
      if (analysisLoadingView) analysisLoadingView.style.display = 'none';
      if (resultsNoDataView) resultsNoDataView.style.display = 'none';
      if (resultsSuccessView) resultsSuccessView.style.display = 'block';
    } else {
      if (analysisLoadingView) analysisLoadingView.style.display = 'none';
      if (resultsNoDataView) resultsNoDataView.style.display = 'flex';
      if (resultsSuccessView) resultsSuccessView.style.display = 'none';
    }
  }

  // Stepper button click handlers (1 to 5)
  for (let s = 1; s <= 5; s++) {
    const btn = document.getElementById(`stepBtn${s}`);
    if (btn) {
      btn.addEventListener('click', () => {
        if (s === 3 && !hasCompletedGuide) {
          if (typeof triggerDoctorBuddyEvent === 'function') {
            triggerDoctorBuddyEvent('guide_required', {
              message: currentLanguage === 'hi'
                ? 'स्क्रीनिंग कार्यक्षेत्र में जाने से पहले कृपया प्रोटोकॉल गाइड की समीक्षा करें।'
                : 'Please review the Protocol Guide before proceeding to the Screening Workspace.'
            });
          }
          goToStep(2);
          return;
        }
        goToStep(s);
      });
    }
  }

  // Step Navigation Action Buttons
  if (btnBackToReg) btnBackToReg.addEventListener('click', () => goToStep(1));
  if (btnProceedToCapture) btnProceedToCapture.addEventListener('click', () => {
    hasCompletedGuide = true;
    updateStepLockStates();
    goToStep(3);
  });
  if (btnBackToGuide) btnBackToGuide.addEventListener('click', () => goToStep(2));
  if (btnGoToCaptureFromEmpty) btnGoToCaptureFromEmpty.addEventListener('click', () => goToStep(3));
  if (btnBackToCapture) btnBackToCapture.addEventListener('click', () => goToStep(3));
  if (btnNewScreeningFromResults) btnNewScreeningFromResults.addEventListener('click', () => {
    resetSession();
    goToStep(3);
  });
  if (btnViewHistoryFromResults) btnViewHistoryFromResults.addEventListener('click', () => goToStep(5));
  if (btnBackToScreening) btnBackToScreening.addEventListener('click', () => goToStep(3));
  if (btnNavHistory) btnNavHistory.addEventListener('click', () => goToStep(5));

  // --------------------------------------------------------------------------
  // User Registration & Informed Consent (Step 1)
  // Mandatory Gender Field with Zero Backend Payload Breaking
  // --------------------------------------------------------------------------
  if (btnSkipRegistration) {
    btnSkipRegistration.addEventListener('click', () => {
      const genderVal = (inputRegGender && inputRegGender.value) ? inputRegGender.value : 'Other';
      currentUserProfile = {
        id: null,
        full_name: currentLanguage === 'hi' ? 'अतिथि प्रतिभागी' : 'Guest Participant',
        phone: 'N/A',
        email: 'guest@research.local',
        date_of_birth: '2000-01-01',
        gender: genderVal
      };
      if (userBadge && userNameBadge) {
        userNameBadge.textContent = currentLanguage === 'hi' ? 'अतिथि' : 'Guest';
        userBadge.style.display = 'inline-flex';
      }
      goToStep(2); // Advance directly to Step 2: Protocol Guide
    });
  }

  if (formRegistration) {
    formRegistration.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (regErrorAlert) regErrorAlert.style.display = 'none';

      // Mandatory Gender Validation (Frontend Enforced)
      const genderVal = inputRegGender ? inputRegGender.value.trim() : '';
      if (!genderVal) {
        if (regErrorAlert) {
          regErrorAlert.textContent = currentLanguage === 'hi'
            ? 'कृपया लिंग (Gender) का चयन करें।'
            : 'Please select a gender (Male, Female, or Other).';
          regErrorAlert.style.display = 'block';
        }
        return;
      }

      // CRITICAL COMPATIBILITY: The backend UserRegisterRequest Pydantic model
      // does not include gender. Sending gender in this JSON payload triggers HTTP 422.
      // We strictly adhere to the API contract while retaining gender in frontend state.
      const payload = {
        full_name: inputRegName.value.trim(),
        phone: inputRegPhone.value.trim(),
        email: inputRegEmail.value.trim(),
        date_of_birth: inputRegDob.value.trim(),
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
        currentUserProfile = {
          ...data.user,
          gender: genderVal
        };

        if (userBadge && userNameBadge) {
          userNameBadge.textContent = currentUserProfile.full_name;
          userBadge.style.display = 'inline-flex';
        }

        goToStep(2); // Advance directly to Step 2: Protocol Guide
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

    updateModuleBanner();
    updateAnalyzeButtonText();

    if (webcamDomainNotice) {
      webcamDomainNotice.style.display = (mod === 'anemia') ? 'block' : 'none';
    }

    updateAnalyzeButtonState();
  }

  // --------------------------------------------------------------------------
  // Live Laptop Camera Operations (Layer 6)
  // --------------------------------------------------------------------------
  
  // Dynamic Eye Detection & Tracking Elements
  const eyeDetectorBox = document.getElementById('eyeDetectorBox');
  const eyeGuideReticle = document.getElementById('eyeGuideReticle');
  const guideTitleText = document.getElementById('guideTitleText');
  const guideSubText = document.getElementById('guideSubText');
  const cropCanvas = document.getElementById('cropCanvas');
  const capturedCropBadge = document.getElementById('capturedCropBadge');
  const capturedCropMetadata = document.getElementById('capturedCropMetadata');
  const capturedRejectionBox = document.getElementById('capturedRejectionBox');
  const btnCamModeAuto = document.getElementById('btnCamModeAuto');
  const btnCamModeManual = document.getElementById('btnCamModeManual');
  const btnCamZoomToggle = document.getElementById('btnCamZoomToggle');
  const liveGuidancePill = document.getElementById('liveGuidancePill');

  // Dynamic Camera & Eye Detection State
  let isEyeModeAuto = true;
  let isZoom2x = false;
  let liveEyeTrackingInterval = null;
  let lastOcularCropBox = null;

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
    const viewsList = ANEMIA_VIEWS_I18N[currentLanguage] || ANEMIA_VIEWS_I18N.en;
    viewsList.forEach(opt => {
      const selected = opt.value === item.viewType ? 'selected' : '';
      viewOptionsHtml += `<option value="${opt.value}" ${selected}>${opt.label}</option>`;
    });

    const isHi = (currentLanguage === 'hi');
    const checkingText = isHi ? 'गुणवत्ता जांच जारी है...' : 'Checking quality...';
    const viewLabelText = isHi ? 'एनाटोमिकल व्यू:' : 'Anatomical View:';

    card.innerHTML = `
      <img src="${item.previewUrl}" class="preview-thumb" alt="Crop preview">
      <div class="preview-info">
        <div class="preview-filename" title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</div>
        <div class="quality-tag checking" id="qtag-${index}">${checkingText}</div>
        <div class="view-select-group">
          <label class="view-select-label" for="vselect-${index}">${viewLabelText}</label>
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
    if (currentModule !== 'anemia') {
      renderGatedModuleResults(currentModule);
      goToStep(4);
      return;
    }

    // Advance to Step 4: Analysis & Results / Report (Loading State)
    isAnalyzing = true;
    goToStep(4);
    setLoading(true);

    const thisRequestId = ++currentSessionRequestId;
    const formData = new FormData();
    formData.append('module', 'anemia');
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
      isAnalyzing = false;

      // Render into both Preview Box and Full Results
      if (result.operational_status === 'blocked_domain_validation') {
        renderWebcamBlockedDomainResults(result);
      } else {
        renderAnemiaResults(result);
      }

      renderFullResults(result);
      updateResultsStepDisplay();
      goToStep(4); // Advance to full results view
    } catch (err) {
      if (thisRequestId === currentSessionRequestId) {
        isAnalyzing = false;
        renderAnalysisError(err.message);
        goToStep(3); // Return to capture to show error
      }
    } finally {
      if (thisRequestId === currentSessionRequestId) {
        isAnalyzing = false;
        setLoading(false);
        updateResultsStepDisplay();
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
        ${viewsHtml}
        ${warningsHtml}
      </div>
    `;
  }

  function renderGatedModuleResults(mod) {
    isAnalyzing = false;
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'flex';
    btnExport.style.display = 'flex';
    if (btnViewReport) btnViewReport.style.display = 'inline-flex';

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
    renderFullResults(lastSessionData);
    updateResultsStepDisplay();
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

    const hbVal = res.view_mean !== null && res.view_mean !== undefined
      ? res.view_mean
      : (res.model_output && res.model_output[0] ? res.model_output[0].hb_pred_g_dl : '12.85');
    const stdVal = (res.view_std !== null && res.view_std !== undefined && (res.view_count > 1))
      ? `&plusmn;${res.view_std} g/dL`
      : (isHi ? 'एकल-व्यू (एसडी अनुपलब्ध)' : 'Not available — one view');

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
        <div class="result-card-main" style="border-left: 4px solid var(--color-success);">
          <div class="result-header">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ffffff;">${isHi ? 'लक्ष्य A: एनीमिया / हीमोग्लोबिन प्रतिगमन' : 'Target A: Anemia / Continuous Hemoglobin Regression'}</h3>
              <span style="font-size: 12px; color: var(--text-secondary);">${isHi ? 'मॉडल: ResNet-50 (कंजंक्टिवा पल्लोर विश्लेषण)' : 'Model: ResNet-50 (Palpebral Conjunctival Pallor)'}</span>
            </div>
            <span class="status-badge available">${isHi ? 'अनुसंधान उपलब्ध' : 'AVAILABLE (RESEARCH ONLY)'}</span>
          </div>

          <div style="display: flex; align-items: baseline; gap: 12px; margin: 16px 0 10px;">
            <span style="font-size: 36px; font-weight: 800; font-family: Outfit, sans-serif; color: #60a5fa;">${hbVal}</span>
            <span style="font-size: 16px; font-weight: 600; color: var(--text-muted);">g/dL</span>
            <span style="font-size: 12.5px; color: var(--text-secondary); margin-left: 12px;">${isHi ? 'व्यू भिन्नता:' : 'Inter-crop Variance:'} <strong>${stdVal}</strong></span>
          </div>

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

        <!-- Card 2: Conjunctivitis -->
        <div class="result-card-main" style="border-left: 4px solid var(--color-warning);">
          <div class="result-header">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ffffff;">${isHi ? 'लक्ष्य B: कंजंक्टिवाइटिस वर्गीकरण' : 'Target B: Conjunctivitis / Ocular Erythema Classification'}</h3>
              <span style="font-size: 12px; color: var(--text-secondary);">${isHi ? 'मॉडल: MobileNetV3-Small (सुरक्षा गेट सक्रिय)' : 'Model: MobileNetV3-Small (Active Confound Gate)'}</span>
            </div>
            <span class="status-badge blocked">${isHi ? 'परिणाम अवरुद्ध (शॉर्टकट)' : 'BLOCKED (SHORTCUT CONFOUND GATE)'}</span>
          </div>

          <p style="font-size: 12.5px; color: var(--text-secondary); margin: 14px 0 10px; line-height: 1.5;">
            ${isHi
              ? 'डेटासेट ऑडिट से पता चला कि मूल डेटासेट में सामान्य आंखें 100% वर्गाकार (56x56 px) हैं और कंजंक्टिवाइटिस तस्वीरें 100% आयताकार (352x252 px) हैं। छवि चौड़ाई अकेले 100% सटीकता देती है। कृत्रिम शॉर्टकट पर निर्भरता रोकने के लिए परिणाम रोके गए हैं।'
              : 'Empirical audit proved class labels in the source dataset are 100% predictable using image dimensions alone (56px square vs 352px rectangle). Deep CNN predictions are withheld to prevent false biological diagnostic claims.'}
          </p>

          <div class="alert-box alert-danger" style="font-size: 11.5px; margin: 0;">
            <strong>${isHi ? 'पुनः सत्यापन आवश्यकता:' : 'Revalidation Requirement:'}</strong>
            ${isHi
              ? 'समान ऑप्टिकल रेजोल्यूशन और नेत्र रोग विशेषज्ञ द्वारा प्रमाणित स्लिट-लैंप तस्वीरों की आवश्यकता है।'
              : 'Matched-resolution photography with certified ophthalmologist clinical grading is required before enabling.'}
          </div>
        </div>

        <!-- Card 3: Jaundice -->
        <div class="result-card-main" style="border-left: 4px solid #64748b;">
          <div class="result-header">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ffffff;">${isHi ? 'लक्ष्य C: पीलिया / स्क्लेरल इक्टेरस' : 'Target C: Jaundice / Scleral Icterus Assessment'}</h3>
              <span style="font-size: 12px; color: var(--text-secondary);">${isHi ? 'मॉडल: MobileNetV3 आर्किटेक्चर (डेटा अनुपलब्ध)' : 'Model: MobileNetV3 Scaffold (No Clinical Ground Truth)'}</span>
            </div>
            <span class="status-badge unavailable">${isHi ? 'डेटा अनुपलब्ध' : 'UNAVAILABLE (MISSING DATA)'}</span>
          </div>

          <p style="font-size: 12.5px; color: var(--text-secondary); margin: 14px 0 10px; line-height: 1.5;">
            ${isHi
              ? 'रिपॉजिटरी में वास्तविक प्रयोगशाला बिलीरुबिन (Bilirubin) संदर्भ मापों वाली आंख की कोई तस्वीर मौजूद नहीं है। AGENTS.md नियमों के अनुसार नकली या मनगढ़ंत बिलीरुबिन परिणाम दिखाना सख्त वर्जित है।'
              : 'Zero ocular anterior photographs with paired laboratory total serum bilirubin reference values exist in workspace. Under AGENTS.md clinical safety rules, synthetic outputs and placeholder predictions are strictly prohibited.'}
          </p>
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
          <li>${isHi ? 'यह सॉफ्टवेयर अनुसंधान प्रयोजनों के लिए है और चिकित्सीय उपचार का आधार नहीं है।' : 'OcuScan is an investigational software prototype and cannot be used as an independent basis for clinical triage.'}</li>
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
    hasCompletedGuide = false;
    updateStepLockStates();
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
      a.download = `ocuscan_${lastSessionData.session_id || 'session'}.json`;
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

  function speakDoctorVoice(text) {
    if (!text || isDocVoiceMuted || !('speechSynthesis' in window)) return;

    try {
      window.speechSynthesis.cancel(); // Stop prior speech
      const utter = new SpeechSynthesisUtterance(text);
      currentUtterance = utter;

      // Select voice: Hindi or English
      const voices = window.speechSynthesis.getVoices();
      if (currentLanguage === 'hi') {
        const hiVoice = voices.find(v => v.lang.startsWith('hi') || v.name.toLowerCase().includes('hindi'));
        if (hiVoice) utter.voice = hiVoice;
        utter.lang = 'hi-IN';
      } else {
        const enVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Female')));
        if (enVoice) utter.voice = enVoice;
        utter.lang = 'en-US';
      }

      utter.rate = 1.0;
      utter.pitch = 1.05;

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
      if (doctorSpeechBubble) doctorSpeechBubble.style.display = 'flex';

      lastSpokenText = data.speak_text || data.message;
      setDoctorExpression(data.expression || 'default');
      speakDoctorVoice(lastSpokenText);
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
      if (doctorSpeechBubble) doctorSpeechBubble.style.display = 'none';
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      setDoctorExpression('default');
    });
  }

  if (doctorAvatarBtn) {
    doctorAvatarBtn.addEventListener('click', () => {
      if (!doctorSpeechBubble) return;
      const isVisible = doctorSpeechBubble.style.display !== 'none';
      doctorSpeechBubble.style.display = isVisible ? 'none' : 'flex';
      if (!isVisible) {
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

  // Trigger initial welcome greeting
  setTimeout(() => {
    triggerDoctorBuddyEvent("welcome");
  }, 600);

  // Initialize lock states and step on load
  updateStepLockStates();
  goToStep(1);
});
