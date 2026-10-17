import os
import sys
import json
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

ARTIFACTS_DIR = r"C:\Users\sarth\.gemini\antigravity\brain\70fdcf38-6d56-4d81-939c-67094dd39db0"
UPLOADED_DIR = os.path.join(ARTIFACTS_DIR, ".user_uploaded")
SCREENSHOTS_DIR = os.path.join(ARTIFACTS_DIR, "browser_screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

test_images = [
    {
        "filename": "media_1791982296276_de397f8c.jpg",
        "label": "Image 1: Direct Macro Eye (Everted Eyelid)",
        "expected_detection": "direct_macro_eye"
    },
    {
        "filename": "media_1791982402907_3b6e56d5.jpg",
        "label": "Image 2: Direct Macro Eye (Sclera + Lower Lid)",
        "expected_detection": "direct_macro_eye"
    },
    {
        "filename": "media_1791982402935_e0e7775e.jpg",
        "label": "Image 3: Portrait / Full Face (Ocular Crop Required)",
        "expected_detection": "auto_crop"
    },
    {
        "filename": "media_1791982472261_5c8f69a1.webp",
        "label": "Image 4: Clinical Close-Up Macro (WebP)",
        "expected_detection": "direct_macro_eye"
    }
]

audit_results = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1400, "height": 1000}, accept_downloads=True)
    page = context.new_page()

    console_logs = []
    page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

    print("Opening OcuScan at http://127.0.0.1:8000 ...")
    page.goto("http://127.0.0.1:8000")
    page.wait_for_load_state("networkidle")

    # Step 1: Language selection -> English
    card_lang_en = page.locator("#cardLangEn")
    if card_lang_en.is_visible():
        card_lang_en.click()
        time.sleep(0.3)
    btn_cont_lang = page.locator("#btnContinueLang")
    if btn_cont_lang.is_visible():
        btn_cont_lang.click()
        time.sleep(0.5)

    # Step 2: Patient Registration
    print("Completing mandatory profile (Name, Age=35, Sex=Male, Consent)...")
    page.fill("#inputRegName", "Clinical Research Subject 01")
    page.fill("#inputRegAge", "35")
    page.select_option("#selectRegGender", "male")
    page.fill("#inputRegPhone", "+91 9876543210")
    page.fill("#inputRegEmail", "research.subject01@eyehealth.ai")
    
    # Check mandatory consent checkboxes
    page.check("#chkConsentResearch")
    page.check("#chkConsentStorage")
    page.check("#chkConsentExternalAi")
    time.sleep(0.5)

    # Click Submit Registration form button
    btn_submit_reg = page.locator("#btnSubmitRegistration")
    print("Submitting profile registration...")
    with page.expect_response(lambda r: "/api/register" in r.url, timeout=10000):
        btn_submit_reg.click()
    
    # Wait for progression to Step 4: Capture Workspace
    page.wait_for_selector("#page4Capture.active", timeout=10000)
    print("Successfully reached Step 4 (Capture Workspace).")
    time.sleep(1.0)

    for idx, item in enumerate(test_images):
        img_path = os.path.join(UPLOADED_DIR, item["filename"])
        print("\n" + "="*70)
        print(f"TESTING REAL IMAGE {idx+1}/4: {item['label']}")
        print(f"Path: {img_path}")

        # Ensure we are on Step 4
        if not page.locator("#page4Capture.active").is_visible():
            page.evaluate("if (window.goToStep) window.goToStep(4);")
            page.wait_for_selector("#page4Capture.active", timeout=10000)
            time.sleep(0.5)

        # Upload image via file input
        file_input = page.locator("#fileInput")
        with page.expect_response(lambda r: "/api/quality-check" in r.url, timeout=20000) as qc_response_info:
            file_input.set_input_files(img_path)

        qc_response = qc_response_info.value
        qc_json = qc_response.json()
        print(f"Quality Check Status: {qc_response.status}")
        print(f"Dimensions: {qc_json.get('dimensions')} -> Crop: {qc_json.get('crop_dimensions')}")
        print(f"Detection: mode={qc_json.get('ocular_detection', {}).get('detection_mode')} | is_macro={qc_json.get('ocular_detection', {}).get('is_macro')}")
        print(f"Quality Gate: {qc_json.get('quality')} | status={qc_json.get('usability_status')}")

        time.sleep(1.0)
        shot_path_step4 = os.path.join(SCREENSHOTS_DIR, f"real_image_{idx+1}_step4_preview.png")
        page.screenshot(path=shot_path_step4)
        print(f"Saved Step 4 screenshot: {shot_path_step4}")

        # Test Before/After preview toggle button if visible
        btn_toggle = page.locator("#btnTogglePreviewCrop")
        if btn_toggle.is_visible():
            print("Testing 'Toggle Eye Crop' Before/After button...")
            btn_toggle.click()
            time.sleep(0.5)
            page.screenshot(path=os.path.join(SCREENSHOTS_DIR, f"real_image_{idx+1}_step4_toggled.png"))
            btn_toggle.click()  # Toggle back to enhanced
            time.sleep(0.3)

        # Click Analyze
        btn_analyze = page.locator("#btnAnalyze")
        print("Clicking Analyze button...")
        with page.expect_response(lambda r: "/api/analyze" in r.url, timeout=30000) as analyze_response_info:
            btn_analyze.click()

        analyze_resp = analyze_response_info.value
        analyze_json = analyze_resp.json()
        session_id = analyze_json.get("session_id")
        print(f"Analyze Status: {analyze_resp.status} | Session ID: {session_id}")

        # Wait for Step 6 (Results)
        page.wait_for_selector(".results-workspace-wrap", timeout=20000)
        time.sleep(1.5)

        shot_path_step6 = os.path.join(SCREENSHOTS_DIR, f"real_image_{idx+1}_step6_results.png")
        page.screenshot(path=shot_path_step6, full_page=True)
        print(f"Saved Step 6 full-page results screenshot: {shot_path_step6}")

        # Extract values displayed on the dashboard DOM
        res_summary_el = page.locator(".res-main-title")
        summary_text = res_summary_el.inner_text() if res_summary_el.is_visible() else "N/A"

        # Model cards
        val_primaries = page.locator(".res-val-primary").all_inner_texts()
        val_subs = page.locator(".res-val-sub").all_inner_texts()
        card_names = page.locator(".res-card-name").all_inner_texts()

        print("\nDisplayed Model Result Cards on Dashboard:")
        cards_data = []
        for cname, vprim, vsub in zip(card_names, val_primaries, val_subs):
            print(f"  • {cname}: {vprim} {vsub}")
            cards_data.append({"card": cname, "value": vprim, "sub": vsub})

        # Image Quality Summary Card
        iq_statuses = page.locator(".iq-item-status").all_inner_texts()
        print("\nDisplayed Image Quality Summary Card Items:")
        for iq in iq_statuses:
            print(f"  • {iq}")

        # AI Explanation
        ai_summary_el = page.locator(".ai-summary-box p, .res-ai-collapse-body p")
        ai_summary_text = ai_summary_el.first.inner_text() if ai_summary_el.count() > 0 else "N/A"
        print(f"\nPersonalized AI Summary: {ai_summary_text}")

        # Test PDF Report Download
        print("Testing Screening PDF report download...")
        btn_pdf_en = page.locator("#btnDownloadPatientPdfEn, #btnDownloadPatientPdf, a[href*='report_type=screening']")
        pdf_path = os.path.join(SCREENSHOTS_DIR, f"real_image_{idx+1}_{session_id}_screening.pdf")
        
        # Download directly via request to verify endpoint integrity
        pdf_url = f"http://127.0.0.1:8000/api/reports/{session_id}/pdf?lang=en"
        pdf_resp = page.request.get(pdf_url)
        print(f"Screening PDF Endpoint ({pdf_url}) Status: {pdf_resp.status}")
        if pdf_resp.status == 200:
            pdf_bytes = pdf_resp.body()
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"Downloaded valid Screening PDF ({len(pdf_bytes)} bytes): {pdf_path}")
        else:
            print(f"WARNING: Failed to download PDF (HTTP {pdf_resp.status})")

        # Record audit item
        record = {
            "image": item["filename"],
            "label": item["label"],
            "session_id": session_id,
            "dimensions_original": qc_json.get("dimensions"),
            "dimensions_crop": qc_json.get("crop_dimensions"),
            "detection": qc_json.get("ocular_detection"),
            "enhancement": qc_json.get("enhancement"),
            "quality_gate": qc_json.get("quality_gate"),
            "model_cards": cards_data,
            "image_quality_summary": iq_statuses,
            "ai_summary": ai_summary_text,
            "screenshots": {
                "step4": shot_path_step4,
                "step6": shot_path_step6
            },
            "pdf_path": pdf_path,
            "pdf_size_bytes": len(pdf_bytes) if pdf_resp.status == 200 else 0
        }
        audit_results.append(record)

        # Navigate back to Step 4 for next image
        page.evaluate("if (window.goToStep) window.goToStep(4);")
        page.wait_for_selector("#page4Capture.active", timeout=10000)
        time.sleep(0.8)

        # Clear existing uploaded image if remove button is present
        btn_remove = page.locator("#btnRemoveImage")
        if btn_remove.is_visible():
            btn_remove.click()
            time.sleep(0.5)

    browser.close()

# Save complete audit report JSON
report_json_path = os.path.join(ARTIFACTS_DIR, "real_images_browser_audit.json")
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print(f"AUDIT COMPLETE. Full audit JSON saved to: {report_json_path}")
print("Console logs recorded:", len(console_logs))
for log in console_logs[-10:]:
    print("  ", log)
