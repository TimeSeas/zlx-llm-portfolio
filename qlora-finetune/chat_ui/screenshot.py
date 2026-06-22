"""Take screenshots of chat interface using Playwright"""
from playwright.sync_api import sync_playwright

import os
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})

    # Screenshot 1: Empty chat interface
    print("Taking screenshot 1: empty interface...")
    page.goto("http://127.0.0.1:5000")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{OUT_DIR}/chat_ui_empty.png", full_page=False)
    print("  Saved: chat_ui_empty.png")

    # Screenshot 2: Type a question and send
    print("Taking screenshot 2: sending message...")
    textarea = page.locator("#user-input")
    textarea.fill("请解释QLoRA微调方法的原理")
    page.locator("#send-btn").click()

    # Wait for response (typing indicator to disappear)
    page.wait_for_timeout(2000)
    # Wait for typing indicator to go away and response to appear
    try:
        page.wait_for_selector("#typing-indicator", state="hidden", timeout=60000)
    except:
        pass
    page.wait_for_timeout(500)

    page.screenshot(path=f"{OUT_DIR}/chat_ui_response1.png", full_page=False)
    print("  Saved: chat_ui_response1.png")

    # Screenshot 3: Send another message (multi-turn)
    print("Taking screenshot 3: second message...")
    textarea = page.locator("#user-input")
    textarea.fill("用Python写一个二分查找算法")
    page.locator("#send-btn").click()

    try:
        page.wait_for_selector("#typing-indicator", state="hidden", timeout=60000)
    except:
        pass
    page.wait_for_timeout(500)

    page.screenshot(path=f"{OUT_DIR}/chat_ui_response2.png", full_page=False)
    print("  Saved: chat_ui_response2.png")

    browser.close()
    print("\nAll screenshots captured!")
