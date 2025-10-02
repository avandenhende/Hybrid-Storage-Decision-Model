from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from itsdangerous import URLSafeSerializer
from io import BytesIO, StringIO
from matplotlib.figure import Figure
from matplotlib.colors import is_color_like
from datetime import datetime, timedelta
from pprint import pprint
import os, base64, uuid, numpy as np, pandas as pd, signal, sys, tempfile, asyncio, requests, math, io

from be.model import get_solar_availability, get_output, annuity, load_env

GUROBI = False


# ------------- Functions ---------------
# Determines if an input value (usually string) can be converted to a float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError or TypeError:
        return False
    
def is_int(value):
    try:
        int(value)
        return True
    except ValueError or TypeError:
        return False
    
def is_hour_apart(date1_str, date2_str):
    # Parse the strings
    fmt = "%Y-%m-%dT%H"
    d1 = datetime.strptime(date1_str, fmt)
    d2 = datetime.strptime(date2_str, fmt)

    # Calculate difference in hours
    diff_hours = abs((d2 - d1).total_seconds()) / 3600
    return diff_hours == 1

def add_hour(date_str):
    fmt = "%Y-%m-%dT%H"
    dt = datetime.strptime(date_str, fmt)
    dt_plus_hour = dt + timedelta(hours=1)
    return dt_plus_hour.strftime(fmt)


def write_cookie(name, value, response: Response):
    token = serializer.dumps(value)
    response.set_cookie(name, token)


def read_cookie(name, request: Request):
    try:
        return serializer.loads(request.cookies.get(name))
    except Exception:
        return None

# Saves a numpy array to a temporary user folder
async def save_array(arr, name, request: Request):
    user_id = read_cookie('session', request)['user_id']
    user_dir = os.path.join(TEMP_FOLDER, user_id)
    os.makedirs(user_dir, exist_ok=True)

    filename = f"{name}.npy"
    filepath = os.path.join(user_dir, filename)
    await run_in_threadpool(np.save, filepath, arr)

    return filepath


# Loads a numpy array from a user's temporary folder
def load_array(filepath):
    if filepath and os.path.exists(filepath):
        return np.load(filepath)
    return None


async def save_csv(df, name, request):
    user_id = read_cookie('session', request)['user_id']
    user_dir = os.path.join(TEMP_FOLDER, user_id)
    os.makedirs(user_dir, exist_ok=True)

    filename = f"{name}.csv"
    filepath = os.path.join(user_dir, filename)
    await run_in_threadpool(df.to_csv, filepath, index_label='Hour of the Year')

    return filename

async def load_csv(filename, request: Request):
    user_id = read_cookie('session', request).get('user_id')
    file_path = os.path.join(TEMP_FOLDER, user_id, filename)

    if not os.path.exists(file_path):
        return throw_error("Sorry an error occured. Please try again.", request)
    
    return await run_in_threadpool(pd.read_csv, file_path)


def throw_error(error_message, request):
    return templates.TemplateResponse('invalid.html', {"request": request, "error_message": error_message})


# -------------- Initiate app ----------------
# Initate upload folders
temp_dir = tempfile.gettempdir()
TEMP_FOLDER = os.path.join(temp_dir, 'tmp')
os.makedirs(TEMP_FOLDER, exist_ok=True)


if getattr(sys, 'frozen', False):
    GID_dir = os.path.dirname(sys.executable)
else:
    GID_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GID_FOLDER = os.path.join(GID_dir, 'processed_gids')
os.makedirs(GID_FOLDER, exist_ok=True)


if getattr(sys, 'frozen', False):
    extDataDir = os.path.join(sys._MEIPASS, 'be')
else:
    extDataDir = os.path.dirname(os.path.abspath(__file__))


app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(extDataDir, 'static')), name="static")
templates = Jinja2Templates(directory=os.path.join(extDataDir, 'templates'))
load_env()
SECRET_KEY = os.getenv('SECRET_KEY')
EIA_API_KEY = os.getenv('EIA_API_KEY')
serializer = URLSafeSerializer(SECRET_KEY)

task_results = {}


# ----------------- Routes ----------------------
@app.get('/', response_class=HTMLResponse)
@app.get('/index', response_class=HTMLResponse)
async def index(request: Request):

    # If this is the first page of a session, define user id and standard variable values
    if request.cookies.get('session_data') is None:
        user_id = str(uuid.uuid4())
        user_dir = os.path.join(TEMP_FOLDER, user_id)
        os.makedirs(user_dir, exist_ok=True)
        session_data = {
            'user_id': user_id,
            'lat': 18.1263,
            'lon': -65.4401,
            'weatherseverity': 11,
            'capexsolar': 1491.64,
            'opexsolar': 0,
            'tilt': 'Latitude',
            'lifetimesolar': 25,
            'efficiencysolar': 0.8592,
            'maxcapsolar': 'None',
            'capexelectric': 347.69,
            'opexelectric': 0,
            'lifetimeelectric': 10,
            'efficiencyelectric': 0.8649,
            'dischargetimeelectric': 4,
            'dischargerateelectric': 0,
            'maxcapelectric': 'None',
            'capexelectrolyzer': 2000,
            'opexelectrolyzer': 0.04,
            'lifetimeelectrolyzer': 30,
            'efficiencyelectrolyzer': 0.6283,
            'maxcapelectrolyzer': 'None',
            'capexhydrogen': 36,
            'lifetimehydrogen': 30,
            'maxcaphydrogen': 'None',
            'capexfuelcell': 2000,
            'opexfuelcell': 0.04,
            'lifetimefuelcell': 30,
            'efficiencyfuelcell': 0.5,
            'maxcapfuelcell': 'None',
            'discountingrate': 0.04,
            'costslack': 10,
            'solarcheck': None,
            'netpowercheck': None,
            'demandcheck': None,
            'electrolyzercheck': None,
            'fuelcellcheck': None,
            'electriccheck': None,
            'hydrogencheck': None,
            'solarcolor': '#FFD700',
            'netpowercolor': '#1f77b4',
            'demandcolor': '#d62728',
            'electrolyzercolor': '#9467bd',
            'fuelcellcolor': '#2ca02c',
            'electriccolor': '#556B2F',
            'hydrogencolor': '#17becf',
            'start': 1,
            'end': 8760
        }
    else:
        session_data = read_cookie('session', request)


    response = templates.TemplateResponse('index.html', {"request": request, **session_data})
    write_cookie('session', session_data, response)
    # Render the index (home) page with initiated valuess
    return response

# Routes to the index page after initiating data
@app.post('/data-initiation', response_class=HTMLResponse)
async def datainitiation(request: Request):

    # Request form and get session data
    form = await request.form()
    session_data = read_cookie('session', request)


    # Format uploaded data
    upload_file = form['upload_data_file']

    if not upload_file or not upload_file.filename.endswith('.csv'):
        return throw_error("Initation data must be uploaded as a CSV file.", request)

    contents = await upload_file.read()

    def read_initiation_csv(contents):
        upload_data = pd.read_csv(StringIO(contents.decode()), header=None)
        new_header = upload_data.iloc[0]  # Grab the first row for the header
        upload_data = upload_data[1:]  # Take the data less the header row
        upload_data.columns = new_header  # Set the header row as the df header
        upload_data = upload_data.reset_index(drop=True) # Reset the index
        columns_keep = ['Label', 'Value']
        columns_drop = [col for col in upload_data.columns if col not in columns_keep]
        upload_data = upload_data.drop(columns_drop, axis=1).set_index('Label').to_dict()['Value']
        return upload_data
    
    try:
        upload_data = await run_in_threadpool(read_initiation_csv, contents)
    except:
        return throw_error("Please upload a valid CSV file, a valid template can be downloaded next to the file input.", request)


    # Parse inputs
    try:
        float_inputs = ['lat', 'lon', 'capexsolar', 'opexsolar', 'lifetimesolar', 'efficiencysolar', 'capexelectric', 'opexelectric', 'lifetimeelectric', 'efficiencyelectric', 'dischargetimeelectric', 'dischargerateelectric', 'capexelectrolyzer', 'opexelectrolyzer', 'lifetimeelectrolyzer', 'efficiencyelectrolyzer', 'capexhydrogen', 'lifetimehydrogen', 'capexfuelcell', 'opexfuelcell', 'lifetimefuelcell', 'efficiencyfuelcell', 'discountingrate', 'costslack']
        for i in float_inputs:
            upload_input = upload_data.get(i)
            upload_input = upload_input.replace(" ", "")
            session_data[i] = float(upload_input) if is_float(upload_input) else session_data[i]

        upload_input = upload_data.get('weatherseverity')
        session_data['weatherseverity'] = min(max(0, math.floor(float(upload_input)*22)), 22) if is_float(upload_input) else session_data[i]
        
        upload_input = upload_data.get('tilt')
        upload_input = upload_input.replace(" ", "")
        session_data['tilt'] = float(upload_input) if is_float(upload_input) else 'Latitude' if upload_input == 'Latitude' else session_data['tilt']

        maxcap_inputs = ['maxcapsolar', 'maxcapelectric', 'maxcapelectrolyzer', 'maxcaphydrogen', 'maxcapfuelcell']
        for i in maxcap_inputs:
            upload_input = upload_data.get(i)
            upload_input = upload_input.replace(" ", "")
            session_data[i] = float(upload_input) if is_float(upload_input) and not pd.isna(upload_input) else 'None'
        
        # Write session data
        response = templates.TemplateResponse('index.html', {"request": request, **session_data})
        write_cookie('session', session_data, response)

    except:
        return throw_error("Your data initiation file was formatted incorrectly. Please check the \"Label\" and \"Value\" columns.", request)


    return response


# Routes to the output page that results from a model call
@app.post('/loading', response_class=HTMLResponse)
async def loading(request: Request):

    # Get form and session data
    form = await request.form()
    session_data = read_cookie('session', request)


    # Read input demand data file
    demand_file = form.get('demand_data_file')

    if demand_file is not None and demand_file.filename != "" and not demand_file.filename.endswith('.csv'):
        return throw_error("Demand data must be uploaded as a CSV file.", request)

    if demand_file is not None and demand_file.filename != "":
        contents = await demand_file.read()
        demand_data = await run_in_threadpool(lambda: pd.read_csv(StringIO(contents.decode()), header=None))

        session_data["demandfilename"] = await save_csv(demand_data, "demandfilename", request)
    elif "demandfilename" in session_data:
        demand_data = await load_csv(session_data["demandfilename"], request)
        demand_data = demand_data.iloc[:, 1]
    else:
        return throw_error("Demand data must be uploaded.", request)
    
    demand_data = demand_data.to_numpy()

    if len(demand_data) != 8760:
        return throw_error("Demand data must be uploaded with 8760 entries.", request)
    
    # Read form inputs
    float_inputs = ['lat', 'lon', 'weatherseverity', 'capexsolar', 'opexsolar', 'lifetimesolar', 'efficiencysolar', 'capexelectric', 'opexelectric', 'lifetimeelectric', 'efficiencyelectric', 'dischargetimeelectric', 'dischargerateelectric', 'capexelectrolyzer', 'opexelectrolyzer', 'lifetimeelectrolyzer', 'efficiencyelectrolyzer', 'capexhydrogen', 'lifetimehydrogen', 'capexfuelcell', 'opexfuelcell', 'lifetimefuelcell', 'efficiencyfuelcell', 'discountingrate', 'costslack']
    for i in float_inputs:
        form_input = form.get(i)
        form_input = form_input.replace(" ", "")
        session_data[i] = float(form_input) if is_float(form_input) else session_data[i]

    form_input = form.get('tilt')
    form_input = form_input.replace(" ", "")
    session_data['tilt'] = float(form_input) if is_float(form_input) else 'Latitude' if form_input == 'Latitude' else session_data['tilt']

    maxcap_inputs = ['maxcapsolar', 'maxcapelectric', 'maxcapelectrolyzer', 'maxcaphydrogen', 'maxcapfuelcell']
    for i in maxcap_inputs:
        form_input = form.get(i)
        form_input = form_input.replace(" ", "")
        session_data[i] = float(form_input) if is_float(form_input) else 'None'
    
    
    response = templates.TemplateResponse('loading.html', {"request": request})
    write_cookie('session', session_data, response)

    asyncio.create_task(run_output(session_data, request))
    task_results[session_data['user_id']] = {'progress': 0, 'message': '', 'complete': False, 'response': None}

    return response


async def run_output(session_data, request: Request):

    def task_callback(pct, msg):
        task_results[session_data['user_id']]['progress'] = pct
        task_results[session_data['user_id']]['message'] = msg

    # Get demand data
    demand_data = await load_csv(session_data["demandfilename"], request)
    demand_data = demand_data.iloc[:, 1]
    demand_data = demand_data.to_numpy()
    

    # Get solar availability and location label from PVWatts API
    location, solar_availability_data, solar_resource_data = await run_in_threadpool(get_solar_availability,
        GID_FOLDER,
        session_data['weatherseverity']/22,
        lat = session_data['lat'],
        lon = session_data['lon'],
        tilt = session_data['tilt'] if is_float(session_data['tilt']) else abs(session_data['lat']),
        losses = (1-session_data['efficiencysolar'])*100,
        task_callback = task_callback
    )
    city, state, country = location
    

    # Format input parameters for the model (other than demand and availability)
    CapExGen = {'Solar': annuity(session_data['capexsolar'], session_data['lifetimesolar'], session_data['discountingrate'])}
    CapExStin = {'Electric':0,
                'Hydrogen':annuity(session_data['capexelectrolyzer'], session_data['lifetimeelectrolyzer'], session_data['discountingrate'])}
    CapExSt = {'Electric': annuity(session_data['capexelectric'], session_data['lifetimeelectric'], session_data['discountingrate']),
            'Hydrogen': annuity(session_data['capexhydrogen'], session_data['lifetimehydrogen'], session_data['discountingrate'])}
    CapExStout = {'Electric':0,
                'Hydrogen':annuity(session_data['capexfuelcell'], session_data['lifetimefuelcell'], session_data['discountingrate'])}
    OpExGen = {'Solar': session_data['opexelectric']}
    OpExStin = {'Electric': session_data['opexelectric'], 'Hydrogen': session_data['opexelectrolyzer']}
    OpExStout = {'Electric': 0, 'Hydrogen': session_data['opexfuelcell']}
    CostSlack = session_data['costslack']
    nin = {'Electric': session_data['efficiencyelectric']**(1/2), 'Hydrogen': session_data['efficiencyelectrolyzer']}
    nout = {'Electric': session_data['efficiencyelectric']**(1/2), 'Hydrogen': session_data['efficiencyfuelcell']}
    d = {'Electric': session_data['dischargerateelectric'], 'Hydrogen': 0}
    duration = session_data['dischargetimeelectric']
    MaxCapGen = {'Solar': float('inf') if session_data['maxcapsolar'] == 'None' else session_data['maxcapsolar']}
    MaxCapStin = {'Electric': float('inf'),
                'Hydrogen': float('inf') if session_data['maxcapelectrolyzer'] == 'None' else session_data['maxcapelectrolyzer']}
    MaxCapSt = {'Electric': float('inf') if session_data['maxcapelectric'] == 'None' else session_data['maxcapelectric'],
                'Hydrogen': float('inf') if session_data['maxcaphydrogen'] == 'None' else session_data['maxcaphydrogen']}
    MaxCapStout = {'Electric': float('inf'),
                'Hydrogen': float('inf') if session_data['maxcapfuelcell'] == 'None' else session_data['maxcapfuelcell']}


    # Run model with input parameters
    task_callback(75, 'Running decision model...')
    LCOE, solarCap, electricCap, electrolyzerCap, hydrogenCap, fuelcellCap, solarProduction, netPower, hydrogenStorageIn, hydrogenStorageOut, electricStorage, hydrogenStorage, demandData = await run_in_threadpool(get_output, demand_data, solar_availability_data, CapExGen, CapExStin, CapExSt, CapExStout, OpExGen, OpExStin, OpExStout, CostSlack, nin, nout, d, duration, MaxCapGen, MaxCapStin, MaxCapSt, MaxCapStout, gurobi=GUROBI)

    netSystemEfficiency = 100 * sum(netPower) / (sum(solarProduction) / session_data['efficiencysolar'])
    waterConsumption = sum(hydrogenStorageIn) * session_data['efficiencyelectrolyzer'] * 0.03 * (18.01528 / 2.016) * (1 / 0.7)
                    # kWh into the electrolyzer * efficiency of the electrolyzer * kg/kWh * kg H2O/kg H2 * water in/water converted (numbers from ***)


    # Create results dataframe
    results = pd.DataFrame.from_dict({
        'Solar Production [kW]': solarProduction,
        'Net System Power [kW]': netPower,
        'Demand [kW]': demandData,
        'Electric Storage [kWh]': electricStorage,
        'Electrolyzer [kW in]': hydrogenStorageIn,
        'Hydrogen [kWh theoretical]': hydrogenStorage,
        'Fuel Cell [kW out]': hydrogenStorageOut,
        'LCOE [$/kWh]': [LCOE] + [None] * 8759,
        'Net System Efficiency [%]': [netSystemEfficiency] + [None] * 8759,
        'Water Consumption [kg H2O]': [waterConsumption] + [None] * 8759,
        'Solar Capacity [kW DC nameplate]': [solarCap] + [None] * 8759,
        'Battery Electric Storage Capacity [kWh]': [electricCap] + [None] * 8759,
        'Electrolyzer Capacity [kW in]': [electrolyzerCap] + [None] * 8759,
        'Hydrogen Storage [kWh theoretical]': [hydrogenCap] + [None] * 8759,
        'Fuel Cell Capacity [kW out]': [fuelcellCap] + [None] * 8759
    })

    # Store output values to the session
    session_data['LCOE'] = round(LCOE, 4)
    session_data['netSystemEfficiency'] = round(netSystemEfficiency, 2)
    session_data['waterConsumption'] = round(waterConsumption, 2)
    session_data['solarCap'] = round(solarCap, 1)
    session_data['electricCap'] = round(electricCap, 1)
    session_data['electrolyzerCap'] = round(electrolyzerCap, 2)
    session_data['hydrogenCap'] = round(hydrogenCap, 1)
    session_data['fuelcellCap'] = round(fuelcellCap, 2)
    session_data['solarProduction'] = await save_array(solarProduction, 'solarProduction', request)
    session_data['netPower'] = await save_array(netPower, 'netPower', request)
    session_data['hydrogenStorageIn'] = await save_array(hydrogenStorageIn, 'hydrogenStorageIn', request)
    session_data['hydrogenStorageOut'] = await save_array(hydrogenStorageOut, 'hydrogenStorageOut', request)
    session_data['electricStorage'] = await save_array(electricStorage, 'electricStorage', request)
    session_data['hydrogenStorage'] = await save_array(hydrogenStorage, 'hydrogenStorage', request)
    session_data['demandData'] = await save_array(demandData, 'demandData', request)
    session_data['downloadresults'] = await save_csv(results, 'results', request)
    print(session_data['downloadresults'])


    # Formats the title based on the PVWatts output
    if city == '' or city is None or city == 'None':
        if state == '' or state is None or state == 'None':
            title = country
        else:
            title = state + ', ' + country
    else:
        title = city + ', ' + state + ', ' + country
    session_data['title'] = title

    
    # Write cookies
    response = templates.TemplateResponse('output.html', {"request": request, **session_data})
    write_cookie('session', session_data, response)
    

    # Renders the output page
    task_callback(100, 'Complete')
    task_results[session_data['user_id']]['complete'] = True
    task_results[session_data['user_id']]['response'] = response

@app.get('/query_output')
async def query_output(request: Request):
    session_data = read_cookie('session', request)

    query = task_results.get(session_data['user_id'], {'progress': 0, 'message': '', 'complete': False, 'response': None})
    return {
        'progress': query['progress'],
        'message': query['message'],
        'complete': query['complete']
    }


@app.get('/output', response_class=HTMLResponse) 
async def output(request: Request):
    session_data = read_cookie('session', request)

    try:
        response = task_results[session_data['user_id']]['response']
    except:
        throw_error('An error occurred when running your model, please try again.', request)

    # Renders the output page
    return response


# Routes to the plot page with a requested matplotlib plot
@app.post('/plot', response_class=HTMLResponse)
async def plot(request: Request):

    # Read session data
    form = await request.form()
    session_data = read_cookie('session', request)


    # Get checkbox inputs
    check_inputs = ['solarcheck', 'netpowercheck', 'demandcheck', 'electrolyzercheck', 'fuelcellcheck', 'electriccheck', 'hydrogencheck']
    for i in check_inputs:
        session_data[i] = form.get(i) != None


    # Get color inputs
    color_inputs = ['solarcolor', 'netpowercolor', 'demandcolor', 'electrolyzercolor', 'fuelcellcolor', 'electriccolor', 'hydrogencolor']
    for i in color_inputs:
        color = form.get(i)
        session_data[i] = color if is_color_like(color) else session_data[i]


    # Take start and end values for the domain of the plot
    int_inputs = ['start', 'end']
    for i in int_inputs:
        form_input = form.get(i)
        session_data[i] = int(form_input) if is_int(form_input) else session_data[i]
    start = session_data['start']
    end = session_data['end']


    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(8, 6))
    fig.set_dpi(200)
    ax1 = fig.subplots()
    lns = []

    ax1.set_xlabel('Hour of the year')
    ax1.set_ylabel('Power [kW]')

    arraynames = ['solarProduction', 'netPower', 'demandData',  'hydrogenStorageIn', 'hydrogenStorageOut', 'electricStorage', 'hydrogenStorage']
    labels = ['Solar Production', 'Net Power', 'Energy Demand', 'Electrolyzer Consumption', 'Fuel Cell Production', 'Electric Storage', 'Hydrogen Storage']
    
    
    # Determine which lines to plot based on the user inputs
    for i in range(0, 5):
        if session_data[check_inputs[i]]:
            arr = load_array(session_data[arraynames[i]])
            templn = ax1.plot(range(start - 1, end), arr[start - 1:end], label = labels[i], color = session_data[color_inputs[i]])
            lns += templn

    ax2 = ax1.twinx()
    ax2.set_ylabel('Storage [kWh]')
    ax2.tick_params(axis='y')

    for i in range(5, 7):
        if session_data[check_inputs[i]]:
            arr = load_array(session_data[arraynames[i]])
            templn = ax2.plot(range(start - 1, end), arr[start - 1:end], label = labels[i], color = session_data.get(color_inputs[i]))
            lns += templn

    ax1.set_title('Hybrid Storage Decision Model', fontsize = 14)
    ax2.legend(lns, [l.get_label() for l in lns], loc = 'lower right')
    fig.tight_layout()


    # Save the figure to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")

    # Embed the results in an HTML format
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    # Write session to cookies
    response = templates.TemplateResponse('plot.html', {"request": request, 'data':data, **session_data})
    write_cookie('session', session_data, response)


    # Return the plot page
    return response


@app.get("/download_results/{filename}")
async def download_results(filename: str, request: Request):
    user_id = read_cookie('session', request).get('user_id')
    file_path = os.path.join(TEMP_FOLDER, user_id, filename)

    if not os.path.exists(file_path):
        return throw_error("Sorry an error occured. Please try again.", request)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )


BA_SUBREGIONS = {
    "AECI": [],
    "AVA": [],
    "AZPS": [],
    "BANC": [],
    "BPAT": [],
    "CHPD": [],
    "CISO": [("PGAE", "Pacific Gas and Electric"), ("SCE", "Southern California Edison"), ("SDGE", "San Diego Gas and Electric"), ("VEA", "Valley Electric Association")],
    "CPLE": [],
    "CPLW": [],
    "DOPD": [],
    "DUK": [],
    "EPE": [],
    "ERCO": [("COAS", "Coast"), ("EAST", "East"), ("FWES", "Far West"), ("NCEN", "North Central"), ("NRTH", "North"), ("SCEN", "South Central"), ("SOUT", "South"), ("WEST", "West")],
    "FMPP": [],
    "FPC": [],
    "FPL": [],
    "GCPD": [],
    "GVL": [],
    "HST": [],
    "IID": [],
    "IPCO": [],
    "ISNE": [("4001", "Maine"), ("4002", "New Hampshire"), ("4003", "Vermont"), ("4004", "Connecticut"), ("4005", "Rhode Island"), ("4006", "Southeast Mass."), ("4007", "Western/Central Mass."), ("4008", "Northeast Mass.")],
    "JEA": [],
    "LDWP": [],
    "LGEE": [],
    "MISO": [("0001", "Zone 1"), ("0004", "Zone 4"), ("0006", "Zone 6"), ("0027", "Zones 2 and 7"), ("0035", "Zones 3 and 5"), ("8910", "Zones 8, 9, and 10")],
    "NEVP": [],
    "NWMT": [],
    "NYIS": [("ZONA", "West"), ("ZONB", "Genesee"), ("ZONC", "Central"), ("ZOND", "North"), ("ZONE", "Mohawk Valley"), ("ZONF", "Capital"), ("ZONG", "Hudson Valley"), ("ZONH", "Millwood"), ("ZONI", "Dunwoodie"), ("ZONJ", "New York City"), ("ZONK", "Long Island")],
    "PACE": [],
    "PACW": [],
    "PGE": [],
    "PJM": [("AE", "Atlantic Electric zone"), ("AEP", "American Electric Power zone"), ("AP", "Allegheny Power zone"), ("ATSI", "American Transmission Systems, Inc. zone"), ("BC", "Baltimore Gas & Electric zone"), ("CE", "Commonwealth Edison zone"), ("DAY", "Dayton Power & Light zone"), ("DEOK", "Duke Energy Ohio/Kentucky zone"), ("DOM", "Dominion Virginia Power zone"), ("DPL", "Delmarva Power & Light zone"), ("DUQ", "Duquesne Lighting Company zone"), ("EKPC", "East Kentucky Power Cooperative zone"), ("JC", "Jersey Central Power & Light zone"), ("ME", "Metropolitan Edison zone"), ("PE", "PECO Energy zone"), ("PEP", "Potomac Electric Power zone"), ("PL", "Pennsylvania Power & Light zone"), ("PN", "Pennsylvania Electric zone"), ("PS", "Public Service Electric & Gas zone"), ("RECO", "Rockland Electric (East) zone")],
    "PNM": [("CYGA", "City of Gallup"), ("Frep", "Freeport"), ("Jica", "Jicarilla Apache Nation"), ("KAFB", "Kirtland Air Force Base"), ("KCEC", "Kit Carson Electric Cooperative"), ("LAC", "Los Alamos County"), ("PNM", "PNM System Firm Load"), ("TSGT", "Tri-State Generation and Transmission")],
    "PSCO": [],
    "PSEI": [],
    "SC": [],
    "SCEG": [],
    "SCL": [],
    "SEC": [],
    "SOCO": [],
    "SPA": [],
    "SRP": [],
    "SWPP": [("CSWS", "AEPW American Electric Power West"), ("EDE", "Empire District Electric Company"), ("GRDA", "Grand River Dam Authority"), ("INDN", "Independence Power & Light"), ("KACY", "Kansas City Board of Public Utilities"), ("KCPL", "Kansas City Power & Light"), ("LES", "Lincoln Electric System"), ("MPS", "KCP&L Greater Missouri Operations"), ("NPPD", "Nebraska Public Power District"), ("OKGE", "Oklahoma Gas and Electric Co."), ("OPPD", "Omaha Public Power District"), ("SECI", "Sunflower Electric"), ("SPRM", "City of Springfield"), ("SPS", "Southwestern Public Service Company"), ("WAUE", "Western Area Power Upper Great Plains East"), ("WFEC", "Western Farmers Electric Cooperative"), ("WR", "Westar Energy")],
    "TAL": [],
    "TEC": [],
    "TEPC": [],
    "TIDC": [],
    "TPWR": [],
    "TVA": [],
    "WACM": [],
    "WALC": [],
    "WAUW": []
}

@app.get('/demand', response_class=HTMLResponse) 
async def demand(request: Request):
    session_data = read_cookie('session', request)
    response = templates.TemplateResponse('demand.html', {"request": request, "ba_subregions": BA_SUBREGIONS, **session_data})
    write_cookie('session', session_data, response)

    # Renders the output page
    return response


ba_global = None
subregion_global = None
annualdemand_global = 'None'

@app.post('/download_demand')
@app.get('/download_demand')
async def download_demand(request: Request):
    global ba_global, subregion_global, annualdemand_global
    
    if request.method == "POST":
        form = await request.form()
        ba = form.get('ba')
        subregion = form.get('subregion', None)
        annualdemand = form.get('annualdemand', 'None')
        
        if subregion is None or subregion == '':
            subregion = 'None'
        
        ba_global = ba
        subregion_global = subregion
        annualdemand_global = annualdemand
    else:
        ba = ba_global
        subregion = subregion_global
        annualdemand = annualdemand_global

    filename = f'demand_data_{ba}_{subregion}.csv'
    filepath = os.path.join(GID_dir, 'EIA_data', filename)

    print(filepath)

    if not os.path.exists(filepath):
        return throw_error("Sorry an error occured. Please try again.", request)

    try:
        # Load CSV
        df = pd.read_csv(filepath, header=None)

        # Convert annualdemand to float
        annualdemand = float(annualdemand)

        # Multiply only numeric columns by annualdemand
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x * annualdemand/8.76)

        # Save to buffer
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        # Return as downloadable CSV
        response = StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response

    except Exception as e:
        return throw_error(f"Error processing file: {str(e)}", request)


# Shutdown route to close the waitress thread
#@app.route('/shutdown', methods=['POST'])
#def shutdown(request: Request):
#    if request.client.host != '127.0.0.1':
#        raise HTTPException(status_code=403, detail="Wrong user")

#    os.kill(os.getpid(), signal.SIGTERM)