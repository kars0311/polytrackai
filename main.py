from playwright.sync_api import sync_playwright
import time

def track():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width': 1280, 'height': 720})
        page = context.new_page()

        try:
            page.goto("https://app-polytrack.kodub.com/0.4.2/")
            page.wait_for_selector("#screen", timeout=30000)

            print("Game loaded. Waiting 10 seconds for game initialization...")
            time.sleep(10)

            print("Starting speed and checkpoint tracking...")

            while True:
                try:
                    # Modified JavaScript to get all digits
                    speed = page.evaluate("""
                        () => {
                            const speedSpans = document.querySelectorAll('.speedometer div span:first-child span');
                            if (speedSpans) {
                                return Array.from(speedSpans).map(span => span.textContent).join('');
                            }
                            return null;
                        }
                    """)

                    if speed is not None:
                        print(f"Current speed: {speed}")
                    else:
                        print("Speed element not found")

                except Exception as e:
                    print(f"Error reading speed: {e}")
                try:

                    # Modified JavaScript to get all digits
                    checkPoint = page.evaluate("""
                        () => {
                            const checkpointSpan = document.querySelector('.checkpoint div span');
                            if (checkpointSpan) {
                                return checkpointSpan.textContent;
                            }
                            return null;
                        }
                    """)

                    if checkPoint is not None:

                        print(f"Checkpoint: {checkPoint}")
                    else:
                        print("Checkpoint element not found")

                except Exception as e:
                    print(f"Error reading checkpoint: {e}")

                time.sleep(0.1)
                try:

                    # Modified JavaScript to get all digits
                    hint = page.evaluate("""
                        () => {
                            const hintElement = document.querySelector('.hint');
                            if (hintElement) {
                                return hintElement.textContent;
                            }
                            return null;
                        }
                    """)

                    if hint is not None:
                        # Check if the hint element has the 'show' class
                        isHintVisible = page.evaluate("""
                            () => {
                                const hintElement = document.querySelector('.hint');
                                return hintElement && hintElement.classList.contains('show');
                            }
                        """)

                        if isHintVisible:
                            print(f"Hint showing: {hint}")
                        else:
                            print("Hint hidden")
                    else:
                        print("Hint element not found")

                except Exception as e:
                    print(f"Error reading checkpoint: {e}")

                try:
                    # Check for time announcer elements
                    time_announcer = page.evaluate("""
                        () => {
                            const trackName = document.querySelector('.time-announcer .track-name');
                            const current = document.querySelector('.time-announcer .current');

                            return {
                                trackName: trackName ? trackName.textContent : null,
                                current: current ? current.textContent : null,
                                isVisible: !!trackName || !!current
                            };
                        }
                    """)

                    if time_announcer['isVisible']:
                        print("track beaten")
                        # if time_announcer['trackName']:
                        #     print(f"Track Name: {time_announcer['trackName']}")
                        # if time_announcer['current']:
                        #     print(f"Current Time: {time_announcer['current']}")

                except Exception as e:
                    print(f"Error reading time announcer: {e}")

                try:
                    # Get the current time from the timer
                    current_time = page.evaluate("""
                        () => {
                            const timeSpans = document.querySelectorAll('.timer .center div p span');
                            if (timeSpans) {
                                return Array.from(timeSpans).map(span => span.textContent).join('');
                            }
                            return null;
                        }
                    """)

                    if current_time is not None:
                        # Format the time nicely with colons and periods
                        time_str = f"{current_time[0:2]}:{current_time[3:5]}.{current_time[6:9]}"
                        print(f"Current Time: {time_str}")
                    else:
                        print("Timer not found")

                except Exception as e:
                    print(f"Error reading timer: {e}")

                time.sleep(0.1)

        except Exception as e:
            print(f"Error: {e}")



        input("Press Enter to close the browser...")

        #next make it check for class hint show but not hint (or hint hide?)


if __name__ == "__main__":
    track()