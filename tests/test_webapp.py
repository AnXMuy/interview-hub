from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://127.0.0.1:8765")
        page.wait_for_load_state("networkidle")
        assert page.locator("#document h1").count() == 1
        assert page.locator("#course-nav .module").count() == 11
        assert page.locator("#course-nav .page-link").count() == 55
        assert page.locator("#code-editor").input_value().startswith('import torch')
        page.screenshot(path="/tmp/llm-workbench.png", full_page=True)
        page.locator("details[data-module='01-attention'] > summary").click()
        page.locator("a[href='#/01-attention/scaled-dot-product-attention']").click()
        page.wait_for_timeout(250)
        assert "Scaled Dot-Product" in page.locator("#document").inner_text()
        assert page.locator("#lesson-indicator").inner_text() == "模块 1 / 6"
        page.locator("#next-lesson").click()
        page.wait_for_timeout(250)
        assert page.locator("#crumb-page").inner_text() == "Multi-Head Attention"
        assert "solutions/pages/01-attention__multi-head-attention.py" in page.locator("#file-name").inner_text()
        page.locator("#prev-lesson").click()
        page.wait_for_timeout(250)
        assert page.locator("#crumb-page").inner_text() == "Scaled Dot-Product Attention"
        explanation_count = int(page.locator("#page-indicator").inner_text().split("/")[1])
        if explanation_count > 1:
            page.locator("#next-page").click()
            page.wait_for_timeout(100)
            assert page.locator("#page-indicator").inner_text().startswith("讲解 2 /")
        page.locator("#run-code").click()
        page.wait_for_timeout(3000)
        print("badge=", page.locator("#result-badge").inner_text(), "output=", page.locator("#result-output").inner_text()[:200])
        assert page.locator("#result-badge").inner_text() == "未实现"
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#show-editor").click()
        assert page.locator("#workspace.editor-view").count() == 1
        browser.close()
    print("web smoke test passed; screenshot: /tmp/llm-workbench.png")


if __name__ == "__main__":
    main()
