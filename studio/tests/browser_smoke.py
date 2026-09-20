"""Optional real-browser check: python -m studio.tests.browser_smoke.

Requires Playwright and Chromium in the developer environment only.
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright

from studio.server import make_server


def main():
    with tempfile.TemporaryDirectory(prefix="vqe-studio-browser-") as directory:
        os.environ["VQE_PENNYLANE_DATA_DIR"] = directory
        os.environ.pop("VQE_TEST_MODE", None)
        control = Path(directory) / "control"
        control.mkdir()
        os.environ["STUDIO_TEST_CONTROL"] = str(control)
        observer_patch = patch(
            "studio.jobs.WORKER_COMMAND",
            [sys.executable, "-m", "studio.tests.worker_probe"],
        )
        observer_patch.start()
        server = make_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}")
                expect(page.locator("#connection")).to_have_text("Python connected")
                expect(page.locator("#fields label")).to_have_count(10)
                expect(page.locator('[name="settings.steps"]')).to_have_value("75")
                expect(page.locator('[name="problem.basis"]')).to_have_value("sto-3g")
                step_size = page.locator('[name="settings.stepsize"]')
                optimizer = page.locator('[name="settings.optimizer"]')
                expect(step_size).to_have_value("0.15")
                optimizer.select_option("RMSProp")
                expect(step_size).to_have_value("0.01")
                step_size.fill("0.3")
                optimizer.select_option("Adam")
                expect(step_size).to_have_value("0.3")
                step_size.fill("")
                page.locator('[name="settings.steps"]').focus()
                expect(step_size).to_have_value("0.15")
                page.locator('[name="problem.molecule"]').select_option("LiH")
                page.locator("#method").select_option("adapt_vqe")
                expect(page.locator('[name="problem.molecule"]')).to_have_value("LiH")
                expect(page.locator('[name="settings.inner_stepsize"]')).to_have_value(
                    "0.2"
                )
                page.locator("#method").select_option("vqe")
                expect(page.locator('[name="problem.molecule"]')).to_have_value("LiH")
                page.locator('[name="problem.molecule"]').select_option("H2")
                page.locator('[name="settings.steps"]').fill("2")
                page.locator("#run").click()
                expect(page.locator(".card")).to_contain_text(
                    "Live computed observations", timeout=120000
                )
                page.locator(".card").get_by_role(
                    "button", name="View", exact=True
                ).click()
                expect(page.locator("#detail-progress")).to_contain_text(
                    "Current energy"
                )
                expect(page.locator(".card progress")).to_have_attribute("value", "50")
                expect(page.locator("#detail-progress progress")).to_have_attribute(
                    "value", "50"
                )
                expect(page.locator("#detail-progress")).to_contain_text("50%")
                (control / "release").touch()
                expect(page.locator("#detail-content")).to_contain_text(
                    "Completed iterations", timeout=120000
                )
                page.locator("#close").click()
                expect(page.locator(".card .completed")).to_have_count(
                    1, timeout=120000
                )
                expect(page.locator(".card svg")).to_have_count(1)
                page.locator(".card").get_by_role(
                    "button", name="View", exact=True
                ).click()
                expect(page.locator("dialog")).to_be_visible()
                expect(page.locator("#detail-content")).to_contain_text(
                    "Completed iterations"
                )
                with page.expect_download() as download:
                    page.get_by_role("button", name="Export JSON").click()
                payload = json.loads(Path(download.value.path()).read_text())
                assert payload["result"]["num_qubits"] == 4
                assert len(payload["result"]["energies"]) == 3
                assert payload["invocation"]["cache_hit"] is False
                page.locator("#detail-content").get_by_role(
                    "button", name="Re-run", exact=True
                ).click()
                expect(page.locator('[name="settings.steps"]')).to_have_value("2")
                expect(page.locator('[name="settings.stepsize"]')).to_have_value("0.15")
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    2, timeout=120000
                )
                first = page.locator(".card").first
                expect(first.locator("dd").filter(has_text="Yes")).to_have_count(1)
                page.reload()
                expect(page.locator(".card .completed")).to_have_count(2)
                page.locator(".compare-choice input").nth(0).check()
                page.locator(".compare-choice input").nth(1).check()
                page.locator("#compare").click()
                expect(page.locator("#detail-title")).to_have_text(
                    "Experiment comparison"
                )
                expect(page.locator("#detail-content polyline")).to_have_count(2)
                expect(page.locator("#detail-content")).to_contain_text(
                    "Requested and resolved configurations are identical"
                )
                page.locator("#close").click()
                page.locator("#method").select_option("adapt_vqe")
                expect(page.locator("#fields label")).to_have_count(10)
                page.locator('[name="settings.max_ops"]').fill("1")
                page.locator('[name="settings.inner_steps"]').fill("2")
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    3, timeout=120000
                )
                adapt = page.locator(".card").filter(
                    has=page.get_by_role("heading", name="H2 / ADAPT-VQE")
                )
                adapt.get_by_role("button", name="View", exact=True).click()
                expect(page.locator("#detail-content")).to_contain_text(
                    "Selected operators"
                )
                page.locator("#detail-content").get_by_role(
                    "button", name="Re-run", exact=True
                ).click()
                expect(page.locator("#method")).to_have_value("adapt_vqe")
                expect(page.locator('[name="settings.max_ops"]')).to_have_value("1")
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    4, timeout=120000
                )
                expect(
                    page.locator(".card").first.locator("dd").filter(has_text="Yes")
                ).to_have_count(1)
                page.locator(".card").first.locator(".compare-choice input").check()
                page.locator("#compare").click()
                expect(page.locator("#detail-content")).to_contain_text(
                    "Iteration counts have different meanings"
                )
                expect(page.locator("#detail-content figure")).to_have_count(2)
                page.screenshot(path="/tmp/vqe-studio-comparison.png", full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                )
                page.locator("#close").click()
                page.set_viewport_size({"width": 1440, "height": 1100})
                # Hold a real calculation, cancel its queued successor, then itself.
                (control / "release").unlink()
                page.locator("#method").select_option("vqe")
                page.locator('[name="settings.steps"]').fill("10")
                page.locator("#run").click()
                active = page.locator(".card").first
                expect(active).to_contain_text(
                    "Live computed observations", timeout=120000
                )
                page.locator('[name="settings.steps"]').fill("11")
                page.locator("#run").click()
                queued = page.locator(".card").first
                expect(queued.locator(".pill")).to_have_text("submitted")
                queued.get_by_role("button", name="Cancel", exact=True).click()
                expect(queued.locator(".pill")).to_have_text("cancelled")
                running = page.locator(".card").filter(
                    has=page.locator(".pill.running")
                )
                running.get_by_role("button", name="View", exact=True).click()
                page.locator("#detail-content").get_by_role(
                    "button", name="Cancel", exact=True
                ).click()
                expect(page.locator("#detail-content > .eyebrow")).to_have_text(
                    "VQE · cancelled"
                )
                page.locator("#close").click()
                page.reload()
                expect(page.locator(".card .cancelled")).to_have_count(2)
                expect(page.locator(".card .completed")).to_have_count(4)
                (control / "release").touch()
                page.locator('[name="settings.steps"]').fill("0")
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    5, timeout=120000
                )
                # Refine the completed zero-step VQE, compare its source, and re-run.
                page.locator(".card").first.get_by_role(
                    "button", name="View", exact=True
                ).click()
                page.get_by_role(
                    "button", name="Refine with VarQITE", exact=True
                ).click()
                expect(page.locator("#method")).to_have_value("varqite")
                expect(page.locator("#initialization")).to_contain_text(
                    "Refining VQE source"
                )
                expect(page.locator('[name="problem.mapping"]')).to_be_disabled()
                expect(page.locator('[name="settings.ansatz"]')).to_be_disabled()
                page.locator('[name="settings.steps"]').fill("2")
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    6, timeout=120000
                )
                page.locator(".card").first.get_by_role(
                    "button", name="View", exact=True
                ).click()
                expect(page.locator("#detail-content")).to_contain_text(
                    "Combined VQE + VarQITE compute runtime"
                )
                expect(page.locator("#detail-content")).to_contain_text(
                    "imaginary-time update"
                )
                with page.expect_download() as download:
                    page.get_by_role("button", name="Export JSON").click()
                refined = json.loads(Path(download.value.path()).read_text())
                assert refined["result"]["initialization"]["source"] == "supplied"
                assert (
                    abs(
                        refined["result"]["energies"][0]
                        - refined["result"]["initialization"]["provenance"]["energy"]
                    )
                    < 1e-9
                )
                page.get_by_role(
                    "button", name="Compare with source", exact=True
                ).click()
                expect(page.locator("#detail-title")).to_have_text(
                    "Experiment comparison"
                )
                expect(page.locator("#detail-content")).to_contain_text(
                    "Combined VQE + VarQITE"
                )
                expect(page.locator("#detail-content")).not_to_contain_text(
                    "Different or incomplete resolved problems"
                )
                page.screenshot(path="/tmp/vqe-studio-refinement.png", full_page=True)
                page.locator("#close").click()
                page.locator(".card").first.get_by_role(
                    "button", name="View", exact=True
                ).click()
                page.locator("#detail-content").get_by_role(
                    "button", name="Re-run", exact=True
                ).click()
                expect(page.locator("#initialization")).to_contain_text(
                    "Refining VQE source"
                )
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    7, timeout=120000
                )
                expect(
                    page.locator(".card").first.locator("dd").filter(has_text="Yes")
                ).to_have_count(1)
                page.get_by_role(
                    "button", name="Start independently", exact=True
                ).click()
                expect(page.locator('[name="problem.mapping"]')).to_be_enabled()
                page.locator('[name="settings.steps"]').fill("0")
                page.locator("#run").click()
                expect(page.locator(".card .completed")).to_have_count(
                    8, timeout=120000
                )
                page.reload()
                expect(page.locator(".card .completed")).to_have_count(8)
                page.screenshot(path="/tmp/vqe-studio-desktop.png", full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                )
                page.screenshot(path="/tmp/vqe-studio-mobile.png", full_page=True)
                assert errors == [], errors
                browser.close()
                print(
                    "Browser smoke passed: VQE/ADAPT/VarQITE, refinement/source comparison, live progress, queued/running cancellation, comparisons, detail/export, restoration, cache reuse, reload, mobile, no JS errors."
                )
        finally:
            (control / "release").touch()
            server.shutdown()
            server.server_close()
            server.jobs.close()
            thread.join()
            observer_patch.stop()


if __name__ == "__main__":
    main()
