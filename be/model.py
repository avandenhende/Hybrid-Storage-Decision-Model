from dotenv import load_dotenv
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, pyomo.environ as pyo, numpy as np, pandas as pd, sys, PySAM.Pvwattsv8 as pv, h5pyd, time, highspy
from pyomo.opt import SolverFactory


def load_env():
    if getattr(sys, 'frozen', False):
        extDataDir = sys._MEIPASS
    else:
        extDataDir = os.getcwd()

    dotenv_path = os.path.join(extDataDir, '.env')
    load_dotenv(dotenv_path=dotenv_path, override=True)


def get_filepath(year):
    try:
        year_int = int(year)
        if 1998 <= year_int <= 2020:
            return f'/nrel/nsrdb/v3/nsrdb_{year_int}.h5'
    except (ValueError, TypeError):
        if year == 'tmy':
            return f'/nrel/nsrdb/v3/tmy/nsrdb_tmy-2020.h5'
        else:
            raise ValueError(f'Input {year!r} is not a valid year or tmy')


def annuity(pv, lifetime, dr):
    return (pv)*((dr*((1+dr)**lifetime))/(((1+dr)**lifetime)-1))

def empty_callback(pct, msg):
    return


def get_gid(lat, lon):
    try:
        with h5pyd.File(get_filepath(1998)) as f:
                print('Accessing HSDS server to find gid')

                dset_coords = f['coordinates'][...]
                tree = cKDTree(dset_coords)
                dist, pos = tree.query((lat, lon))
                print(f'Gid found {dist} degrees away from ({lat}, {lon}): {pos}')
        return pos
    except:
        print('HSDS server access failed when finding gid, trying again')
        time.sleep(1)
        return get_gid(lat, lon)


def process_year_ranking(year: int, gid: int):
    # Fetch and process data from HSDS for a given year.
    try:
        with h5pyd.File(get_filepath(year)) as f:
            print(f'Accessing HSDS server to rank year {year}')

            dni = f['dni']
            scale = dni.attrs['psm_scale_factor']
            dni = dni[:, gid]
            dni = dni[::2] / scale
            if len(dni) > 8760:
                dni = np.concatenate((dni[:1416], dni[1440:]))
        
        summeddni = np.add.reduceat(dni, np.arange(0, len(dni), 24))
        ranking_factor = ((summeddni.max() + summeddni.min()) / 2 ) - np.std(summeddni)
        print(f'Ranking factor for year {year} is {ranking_factor}')

        return year, ranking_factor

    except Exception as e:
        print(f'HSDS server access failed when ranking year {year}, trying again')
        time.sleep(1)
        return process_year_ranking(year, gid)


def process_year_solar_availability(year, gid: int, lat, lon, tilt, losses, task_callback = empty_callback):
    # Fetch and process data from HSDS for a given year.
    try:
        with h5pyd.File(get_filepath(year)) as f:
            print(f'Accessing HSDS server for solar availability processing for selected year {year}')

            meta = f['meta'][gid]
            time_index = pd.to_datetime(f['time_index'][...].astype(str))
            is_feb_29 = (time_index.month == 2) & (time_index.day == 29)
            time_index = time_index[~is_feb_29]
            if year != 'tmy':
                time_index = time_index[::2]

            
            city = meta['urban'].decode('utf-8')
            state = meta['state'].decode('utf-8')
            country = meta['country'].decode('utf-8')


            solar_resource_data = {'lat': float(meta['latitude']), 'lon': float(meta['longitude']), 'tz': int(meta['timezone']), 'elev': float(meta['elevation'])}
            solar_resource_data['year'] = time_index.year
            solar_resource_data['month'] = time_index.month
            solar_resource_data['day'] = time_index.day
            solar_resource_data['hour'] = time_index.hour
            solar_resource_data['minute'] = time_index.minute

            dataset_keys = ['dni', 'dhi', 'air_temperature', 'wind_speed']
            resource_data_keys = ['dn', 'df', 'tdry', 'wspd']


            def get_dataset(i, f, gid, dataset_keys):
                print(f"Getting {dataset_keys[i]} from HSDS")
                dset = f[dataset_keys[i]]
                temp_data = dset[:, gid]
                if year != 'tmy':
                    temp_data = temp_data[::2]
                temp_data = temp_data / dset.attrs['psm_scale_factor']
                if len(temp_data) > 8760:
                    temp_data = np.concatenate((temp_data[:1416], temp_data[1440:]))
                temp_data = np.roll(temp_data, int(meta['timezone']))

                return (i, temp_data.tolist())

            progress = 50
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_dataset = {executor.submit(get_dataset, i, f, gid, dataset_keys): i for i in range(0, 4)}

                for future in as_completed(future_to_dataset):
                    i, temp_data = future.result(timeout = 20)
                    solar_resource_data[resource_data_keys[i]] = temp_data
                    progress = progress + 25/4
                    task_callback(progress, "Processing solar availability data for selected weather severity...")

        pv_model = pv.new()#default("PVWattsNone")
        pv_model.SolarResource.solar_resource_data = solar_resource_data
        pv_model.SystemDesign.array_type = 0
        pv_model.SystemDesign.azimuth = 180*(lat>0)
        pv_model.SystemDesign.losses = losses
        pv_model.SystemDesign.system_capacity = 1
        pv_model.SystemDesign.tilt = tilt
        
        pv_model.execute()
        print('PVWatts mode run successfully')

        return year, city, state, country, np.array(pv_model.Outputs.ac)/1000, solar_resource_data

    except Exception as e:
        print('HSDS server access failed for solar availability call, trying again')
        print(f'Failed with exception: {e}')
        time.sleep(1)
        return process_year_solar_availability(year, gid, lat, lon, tilt, losses)

# Calls the NREL PVWatts API library to access hourly solar data based on location, tilt, and losses 
def get_solar_availability(GID_FOLDER, weatherseverity, lat, lon, tilt = 0, losses = 14.08, task_callback = empty_callback):
    task_callback(0, 'Processing historical weather data, this may take a few minutes...')
    print('Processing historical weather data, this may take a few minutes')

    gid = get_gid(lat, lon)
    task_callback(5, 'Processing historical weather data, this may take a few minutes...')
    filepath = os.path.join(GID_FOLDER, str(gid) + '.npy')


    if weatherseverity == 'tmy':
        chosen_year = 'tmy'

    elif os.path.exists(filepath):
        rankings = np.load(filepath)
        print('Historical weather data loaded successfully')
        # weatherseverity is 0-1, 0 is a good year, 1 is a bad year
        chosen_year = rankings[int((1-weatherseverity)*(len(rankings)-1))]
    
    else:
        years = range(1998, 2021)
        rankings = []

        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_year = {executor.submit(process_year_ranking, year, gid): year for year in years}

            progress = 0
            for future in as_completed(future_to_year):
                year, ranking_factor = future.result(timeout = 20)
                rankings.append((ranking_factor, year))
                progress = progress + 45/23
                task_callback(progress, 'Processing historical weather data, this may take a few minutes...')
        
        rankings = sorted(rankings)
        rankings = np.array([val for _, val in rankings])
        np.save(filepath, rankings)
        print('Historical weather data ranked successfully')

        # weatherseverity is 0-1, 0 is a good year, 1 is a bad year
        chosen_year = rankings[int((1-weatherseverity)*(len(rankings)-1))]
    
    task_callback(50, 'Processing solar availability data for selected weather severity...')
    year, city, state, country, result, solar_resource_data = process_year_solar_availability(chosen_year, gid, lat, lon, tilt, losses, task_callback = task_callback)
    print('Solar availability data has been processed successfully')

    return (city, state, country), result, solar_resource_data  # choose one results

# Runs a capacity expansion model to calculate the optimal installed capacity based on user inputs
def get_output(demand_data, solar_availability_data, CapExGen, CapExStin, CapExSt, CapExStout, OpExGen, OpExStin, OpExStout, CostSlack, nin, nout, d, duration, MaxCapGen, MaxCapStin, MaxCapSt, MaxCapStout, gurobi = False):

    # Define set variables
    T = 8760
    G = 1
    Gens = ['Solar']
    S = 2
    Stor = ['Electric', 'Hydrogen']

    # Initialize Parameters
    D = demand_data.flatten() # kW
    Availability = solar_availability_data.reshape(1, len(solar_availability_data)) # Unitless


    A = {
        (Gens[i], j):Availability[i,j]
        for i in range(G)
        for j in range(T)
        }

    # Create model
    m = pyo.ConcreteModel()

    # Define sets
    m.T = pyo.Set(initialize=list(range(T)))
    m.G = pyo.Set(initialize=Gens)
    m.S = pyo.Set(initialize=Stor)

    # Define parameters
    m.D = pyo.Param(m.T, initialize = D)
    m.A = pyo.Param(m.G, m.T, initialize = A)
    m.CapExGen = pyo.Param(m.G, initialize = CapExGen)
    m.CapExStin = pyo.Param(m.S, initialize = CapExStin)
    m.CapExSt = pyo.Param(m.S, initialize = CapExSt)
    m.CapExStout = pyo.Param(m.S, initialize = CapExStout)
    m.OpExGen = pyo.Param(m.G, initialize = OpExGen)
    m.OpExStin = pyo.Param(m.S, initialize = OpExStin)
    m.OpExStout = pyo.Param(m.S, initialize = OpExStout)
    m.CostSlack = pyo.Param(initialize = CostSlack)
    m.nin = pyo.Param(m.S, initialize = nin)
    m.nout = pyo.Param(m.S, initialize = nout)
    m.d = pyo.Param(m.S, initialize = d)
    m.duration = pyo.Param(initialize = duration)
    m.MaxCapGen = pyo.Param(m.G, initialize = MaxCapGen)
    m.MaxCapStin = pyo.Param(m.S, initialize = MaxCapStin)
    m.MaxCapSt = pyo.Param(m.S, initialize = MaxCapSt)
    m.MaxCapStout = pyo.Param(m.S, initialize = MaxCapStout)

    # Define variables
    m.C = pyo.Var(m.G, domain = pyo.NonNegativeReals)
    m.Cs = pyo.Var(m.S, domain = pyo.NonNegativeReals)
    m.Csin = pyo.Var(m.S, domain = pyo.NonNegativeReals)
    m.Csout = pyo.Var(m.S, domain = pyo.NonNegativeReals)
    m.P = pyo.Var(m.G, m.T, domain = pyo.NonNegativeReals)
    m.Pin = pyo.Var(m.S, m.T, domain = pyo.NonNegativeReals)
    m.Pout = pyo.Var(m.S, m.T, domain = pyo.NonNegativeReals)
    m.St = pyo.Var(m.S, m.T, domain = pyo.NonNegativeReals)
    m.Slack = pyo.Var(m.T, domain = pyo.NonNegativeReals)

    # Define objective function
    def ObjRule(m):
        objrule = sum(m.C[g]*m.CapExGen[g] for g in m.G)
        objrule += sum(m.Cs[s]*m.CapExSt[s] for s in m.S)
        objrule += sum(m.Csin[s]*m.CapExStin[s] for s in m.S)
        objrule += sum(m.Csout[s]*m.CapExStout[s] for s in m.S)
        objrule += sum(m.P[g, t]*m.OpExGen[g] for g in m.G for t in m.T)
        objrule += sum(m.Pin[s, t]*m.OpExStin[s] for s in m.S for t in m.T)
        objrule += sum(m.Pout[s, t]*m.OpExStout[s] for s in m.S for t in m.T)
        objrule += sum(m.Slack[t]*m.CostSlack for t in m.T)
        objrule += sum((m.Cs['Hydrogen'] - m.St['Hydrogen', t])*0.0001 for t in m.T)
        return objrule
    m.obj = pyo.Objective(rule = ObjRule)

    #Define constraints
    def demand_constr(m, t):
        return sum(m.P[g, t] for g in m.G) + sum(m.Pout[s, t] - m.Pin[s, t] for s in m.S) + m.Slack[t] == D[t]
    m.Demand = pyo.Constraint(m.T, rule=demand_constr)

    def gen_capacity_constr(m, g, t):
        return m.P[g, t] <=  m.C[g]*A[g, t]
    m.Gen_Capacity = pyo.Constraint(m.G, m.T, rule=gen_capacity_constr)

    def gen_maxcap_constr(m, g):
        return m.C[g] <= m.MaxCapGen[g]
    m.Gen_Maxcap = pyo.Constraint(m.G, rule=gen_maxcap_constr)

    def stor_capacity_constr(m, s ,t):
        return m.St[s, t] <= m.Cs[s]
    m.Stor_Capacity = pyo.Constraint(m.S, m.T, rule=stor_capacity_constr)

    def stor_maxcap_constr(m, s):
        return m.Cs[s] <= m.MaxCapSt[s]
    m.Stor_Maxcap = pyo.Constraint(m.S, rule=stor_maxcap_constr)

    def storin_capacity_constr(m, s, t):
        return m.Pin[s, t] <= m.Csin[s]
    m.Storin_Capacity = pyo.Constraint(m.S, m.T, rule=storin_capacity_constr)

    def storin_maxcap_constr(m, s):
        return m.Csin[s] <= m.MaxCapStin[s]
    m.Storin_Maxcap = pyo.Constraint(m.S, rule=storin_maxcap_constr)

    def electricstorin_maxcap_constr(m):
        return m.Csin['Electric']*m.duration <= m.Cs['Electric']
    m.ElectricStorin_Maxcap = pyo.Constraint(rule=electricstorin_maxcap_constr)

    def storout_capacity_constr(m, s, t):
        return m.Pout[s, t] <= m.Csout[s]
    m.Storout_Capacity = pyo.Constraint(m.S, m.T, rule=storout_capacity_constr)

    def storout_maxcap_constr(m, s):
        return m.Csout[s] <= m.MaxCapStout[s]
    m.Storout_Maxcap = pyo.Constraint(m.S, rule=storout_maxcap_constr)

    def electricstorout_maxcap_constr(m):
        return m.Csout['Electric']*m.duration <= m.Cs['Electric']
    m.ElectricStorout_Maxcap = pyo.Constraint(rule=electricstorout_maxcap_constr)

    def stor_flow_constr(m, s, t):
        return (1-m.d[s])*m.St[s, (t-1)%T] + m.Pin[s, t]*m.nin[s] - m.Pout[s, t]/m.nout[s] == m.St[s, t]
    m.Stor_Flow = pyo.Constraint(m.S, m.T, rule=stor_flow_constr)

    # Solve using HiGHS (would love to use Gurobi, but licenses are expensive)
    solver = SolverFactory('highs')
    if gurobi:
        solver = SolverFactory('gurobi')
    results = solver.solve(m ,tee=False)

    # Collect model results
    total_cost = results['Problem'][0]['Upper bound']
    total_production = sum(pyo.value(m.D[t]) for t in m.T) - sum(pyo.value(m.Slack[t]) for t in m.T)
    total_production = float(total_production)
    LCOE = total_cost/total_production if total_production != 0 else 0
    solarCap = pyo.value(m.C[Gens[0]])
    electricCap = pyo.value(m.Cs[Stor[0]])
    electrolyzerCap = pyo.value(m.Csin[Stor[1]])
    hydrogenCap = pyo.value(m.Cs[Stor[1]])
    fuelcellCap = pyo.value(m.Csout[Stor[1]])

    solarProduction = np.array([ pyo.value(m.P[Gens[0],t]) for t in m.T ]).flatten()
    electricStorageIn = np.array([ pyo.value(m.Pin[Stor[0],t]) for t in m.T ]).flatten()
    electricStorageOut = np.array([ pyo.value(m.Pout[Stor[0],t]) for t in m.T ]).flatten()
    electricStorage = np.array([ pyo.value(m.St[Stor[0],t]) for t in m.T ]).flatten()
    hydrogenStorageIn = np.array([ pyo.value(m.Pin[Stor[1],t]) for t in m.T ]).flatten()
    hydrogenStorageOut = np.array([ pyo.value(m.Pout[Stor[1],t]) for t in m.T ]).flatten()
    hydrogenStorage = np.array([ pyo.value(m.St[Stor[1],t]) for t in m.T ]).flatten()

    # Yes this naming can be confusing, 1 kWh in 1 hour = 1 kW, in this case energy/1 = power (fortunately or unfortunately) so netEnergy = netPower
    netEnergy = np.add(solarProduction, electricStorageOut)
    netEnergy = np.add(netEnergy, hydrogenStorageOut)
    netEnergy = np.subtract(netEnergy, electricStorageIn)
    netEnergy = np.subtract(netEnergy, hydrogenStorageIn).flatten()

    return LCOE, solarCap, electricCap, electrolyzerCap, hydrogenCap, fuelcellCap, solarProduction, netEnergy, hydrogenStorageIn, hydrogenStorageOut, electricStorage, hydrogenStorage, D

# Depreciated, too lazy to fix and not necessary
if __name__ == "__main__":
    lat = 18.1263
    lon = -65.4401

    locations, results = get_solar_availability(lat, lon)
    print(locations)

    out = pd.DataFrame.from_dict(results)
    out.to_csv('test.csv', index = False)