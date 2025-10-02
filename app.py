import os, sys
os.environ["AWS_S3_GATEWAY"] = "http://s3.us-west-2.amazonaws.com"
os.environ["AWS_S3_NO_SIGN_REQUEST"] = "1"
os.environ["HS_ENDPOINT"] = "http://localhost:5101"
os.environ["HS_BUCKET"] = "nrel-pds-hsds"

if getattr(sys, 'frozen', False):
    MAIN_DIR = os.path.dirname(sys.executable)
else:
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

#os.environ["H5PYD_CFG"] = os.path.join(MAIN_DIR, '.hscfg')


import webview, threading, sys, shutil, psutil, time, logging, python_multipart, subprocess
from uvicorn import run

from be.start import app, TEMP_FOLDER

def start_flask():
    try:
        run(app, port = 8000)
    except Exception as e:
        print(f'Backend startup failed: {str(e)}')
        sys.exit(1)


def kill_lingering_hsds():
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and isinstance(cmdline, list) and any('hsds' in arg.lower() for arg in cmdline):
                print(f"Found lingering HSDS process (PID: {proc.info['pid']}), killing it")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def start_hsds_server():
    HSDS_LOG = os.path.join(MAIN_DIR, 'hs.log')

    # Clear old logs
    if os.path.exists(HSDS_LOG):
        os.remove(HSDS_LOG)
        print("hs.log cleared")


    # Add libraries to path
    libs_path = os.path.join(MAIN_DIR, 'hsds_runtime', 'python')
    sys.path.insert(0, libs_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{libs_path}"


    os_name = sys.platform
    if os_name.lower() == 'darwin':
        py_exe = os.path.join(MAIN_DIR, 'hsds_runtime', 'python3', 'bin', 'python3')

        p = subprocess.Popen(
            [py_exe, os.path.join(MAIN_DIR, 'hsds_runtime', 'hsds'),
            f'--logfile={HSDS_LOG}'],
            cwd=os.path.dirname(MAIN_DIR),
            env = env
        )
    else:
        py_exe = os.path.join(MAIN_DIR, 'hsds_runtime', 'python', 'python.exe')

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        p = subprocess.Popen(
            [py_exe, os.path.join(MAIN_DIR, 'hsds_runtime', 'hsds.exe')],
            startupinfo=startupinfo, env=env
        )
    print('HSDS thread started')

    ready_event = threading.Event()

    # Monitor server output for readiness
    def monitor_log():
        print("Monitoring hs.log for readiness...")
        for _ in range(120):  # Try for up to 60 seconds
            try:
                with open(HSDS_LOG, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "READY" in line.upper():
                            print("[HSDS] Ready signal detected")
                            ready_event.set()
                            return
            except FileNotFoundError:
                pass  # File may not exist yet
            time.sleep(0.5)

    threading.Thread(target=monitor_log, daemon=True).start()

    if not ready_event.wait(timeout=60):
        raise RuntimeError("HSDS failed to start within timeout")
    else:
        print("HSDS server is ready")
        logging.getLogger().setLevel(logging.WARNING)

    return p


def on_close(process):
    
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
    print("Temp folders removed")

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
    
    sys.exit(0)


def start_app(process):

    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()


    primary_screen = webview.screens[0]
    screen_width = primary_screen.width
    screen_height = primary_screen.height


    webview.settings["ALLOW_DOWNLOADS"] = True
    window = webview.create_window(
        'Hybrid Storage Decision Model',
        url = 'http://localhost:8000', 
        width = screen_width,
        height = screen_height,
        resizable = True,
        fullscreen = False,
        frameless = False,
        draggable = True,
        text_select = True
    )
    
    window.events.closed += lambda: on_close(process)

    webview.start(debug=False)


hsds_process = None
def splash_loop(splash):
    global hsds_process

    kill_lingering_hsds()
    hsds_process = start_hsds_server()

    splash.destroy()


if __name__ == '__main__':
    splash = webview.create_window(
        "Loading...",
        html = """
        <div style="display: flex; justify-content: center; align-items: center; height: 140px">
            <h2 style="font-family: Arial, sans-serif; line-height: 1; text-align: center; font-size: 48px;">
            HS<br>DM
            </h2>
        </div>
        """,
        width = 200,
        height = 200,
        resizable=False,
        frameless=True
    )
    # webview.start runs splash_loop in the main thread after window is ready
    webview.start(func=splash_loop, args=(splash,))

    start_app(hsds_process)
    #kill_lingering_hsds()
    #process = start_hsds_server()
    #start_app(process)


# User manual FAQ and Acknowledgements
# User manual licenses (and licenses file)
# Updated mac version for ARM



# Ground services/facilities - 
# UTV usage (min/max fuel usage per run, power output, etc.)
# 
# Framework or UTV design and requirements

# 
# Fed ex truck hydrogen analogy
#
# Level 2 chargers maintenance