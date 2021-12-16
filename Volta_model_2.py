"""
Created on Wed Nov 24 13:28:27 2021

@author: aow001 based on https://github.com/JazminZatarain/MUSEH2O/blob/main/susquehanna_model.py
"""
#Lower Volta River model

#import Python packages
import os
import numpy as np

#import rbf_functions 
#Install using pip and restart kernel if unavailable
import utils #install using:  pip install utils
from numba import njit #pip install numba #restart kernel (Ctrl + .)

# define path
def create_path(rest): 
    my_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(my_dir, rest))


class VoltaModel:
    gammaH20 = 1000.0 #density of water- 1000 kg/m3
    GG = 9.81 #acceleration due to gravity- 9.81 m/s2
    n_days_in_year = 365 #days in a year                                        #(leap year??***)
    
    #initial conditions
    def __init__(self, l0_Akosombo, d0, n_years, rbf, historic_data=True):
        """

        Parameters
        ----------
        l0_Akosombo : float
                     initial condition Akosombo
        d0 : int
             initial start date
        n_years : int
        rbf : callable
        historic_data : bool, optional
                        if true use historic data, if false use stochastic
                        data
        """

        self.init_level = l0_Akosombo  # historical average water level_feet
        self.day0 = d0 # start day  
        
        self.log_level_release = False                                         #(there's a FIXME here in the Sasquehana***)
        
        # variables from the header file (#not sure what this is but it's necessary***)
        self.input_min = []
        self.input_max = []
        self.output_max = []
        self.rbf = rbf
        
        # log level / release                                   #***add rarea for inundated area ?
        self.blevel_Ak = [] # water level at Akosombo
        self.rirri = []     # water released for irrigation
        self.rflood = []    # flood release- excess water above 276ft @ Akosombo 
        self.renv = []      # environmental flow release
        
        # variables for historical record (1965- 2016) and 1000 year simulation horizon, using extended dataset
        self.n_years = n_years #depending on the dataset (historical or simulation)
        self.time_horizon_H = self.n_days_in_year * self.n_years
        self.hours_between_decisions = 24  # daily/24-hours decision time step but can be reduced so you have subdaily decisions
        self.decisions_per_day = int(24 / self.hours_between_decisions) # one decision a day
        self.n_days_one_year = 365      #(leap year??- check if this is accounted for later**** Also, isnt this repeted? ee line 26)
        
        # Constraints for the reservoirs
        self.min_level_Akosombo = 240  # ft _ min level for hydropower generation and (assumed) min level for irrigation intakes
        self.spill_crest = 236  # ft _ spillway crest at Akosombo dam (#not really a constraint since its even lower than min level for turbines)
        self.flood_warn_level = 276 #ft _ flood warning level where spilling starts (absolute max is 278ft)
        
        
        #Akosombo Characteristics
        self.lsv_rel = utils.loadMatrix (
            create_path("./Data/Akosombo_xtics/1.Level-Surface area-Volume/lsv_Ak.txt"), 3, 6 
            )  # level (ft) - Surface (acre) - storage (acre-feet) relationships
        self.turbines = utils.loadMatrix(
            create_path("./Data/Akosombo_xtics/2.Turbines/turbines_Ak.txt"), 3, 1
            )  # Max capacity (cfs) - min capacity (cfs) - efficiency of Akosombo plant turbines
        self.spillways = utils.loadMatrix(
            create_path("./Data/Akosombo_xtics/4.Spillways/spillways_Ak.txt"), 3, 1
            ) #level (ft) - max release (cfs) - min release (cfs) for level  276 ft (should this be 236?***)
                
        #Kpong Characteristics
        self.turbines_Kp = utils.loadMatrix(
            create_path("./Data/Kpong_xtics/2.Turbine/turbines_Kp.txt"), 2, 1 
            ) # Turbine capacity (cfs) - efficiency of Kpong
        
        
        self.historic_data = historic_data
        if historic_data:
            self.load_historic_data()
            self.evaluate = self.evaluate_historic
        else:
            self.load_stochastic_data()
            self.evaluate = self.evaluate_mc
           
           
        #Objective parameters 
        self.annual_power = 4415
        # annual hydropower requirement (GWh)
        self.daily_power = 6
        # minimum (firm) daily power requirement (GWh)
        self.annual_irri = 1342
        # annual irrigation demand (cfs) (=38m3/s rounded to whole number)
        self.inundated_area = 81224 
        # monthly flowrate above which flooding occurs (cfs) (=2300m3/s rounded up) for objective that limits all flooding
        self.flood_protection = 276
        # reservoir level above which spilling is triggered. For objective that limits area of inundation during flooding
        self.clam_eflows = 330 #placeholder, to be corrected actually a range from Nov to March (50 to 330m3/s)
        # low flows required in November to March (m3/s)
        
        # standardization of the input-output of the RBF release curve      (#***not clear)
        self.input_max.append(self.n_days_in_year * self.decisions_per_day - 1) #365*4 -1
        self.input_max.append(120)                                              #not sure what this is*** 
        
        self.output_max.append(utils.computeMax(self.annual_irri))         #is this for only consumptive water uses
        self.output_max.append(787416)
        # max release = total turbine capacity + spillways @ max storage (56616 + 730800= 787416 cfs)

    def load_historic_data(self):   #ET, inflow, Akosombo tailwater, Kpong fixed head
        self.evap_Ak = utils.loadMultiVector(
            create_path("./Data/Historical_data/Akosombo_ET_1981-2013.txt"),
            self.n_years, self.n_days_one_year
            ) #evaporation losses @ Akosombo 1981-2013 (inches per day)
        self.inflow_Ak = utils.loadMultiVector(
            create_path("./Data/Historical_data/Inflow(cfs)_to_Akosombo_1981-2013.txt"),
            self.n_years, self.n_days_one_year
            )  # inflow, i.e. flows to Akosombo   (cfs) 1981-2013     
        self.tailwater = utils.loadMultiVector(
            create_path("./Data/Historical_data/3.Tailwater/tailwater_Ak.txt"), 
            self.n_years, self.n_days_one_year
            ) # historical tailwater level @ Akosombo (ft) 1981-2013
        self.fh_Kpong = utils.loadMultiVector(create_path(
            "./Data/Historical_data/1.Fixed_head/fh_Kpong.txt"),
            self.n_years, self.n_days_one_year
            ) # historical fixed head Kpong (ft) 1984-2012
        
        
    def load_stochastic_data(self):   # stochastic hydrology###   for now, no stochastic data
        self.evap_Ak = utils.loadMultiVector(
            create_path("./Data/Stochastic_data/Akosombo_ET_stochastic.txt"),
            self.n_years, self.n_days_one_year
            ) #evaporation losses @ Akosombo_ stochastic data (inches per day)
        self.inflow_Ak = utils.loadMultiVector(
            create_path("./Data/Stochastic_data/Inflow(cfs)_to_Akosombo_stochastic.txt"),
            self.n_years, self.n_days_one_year
        )  # inflow, i.e. flows to Akosombo_stochastic data     
        self.tailwater_Ak = utils.loadMultivector(
            create_path("./Data/Stochastic_data/3.Tailwater/tailwater_Ak.txt"), 
            self.n_years, self.n_days_one_year
            ) # historical tailwater level @ Akosombo (ft) 1981-2013
        self.fh_Kpong = utils.loadMultivector(create_path(
            "./Data/Stochastic_datas/1.Fixed_head/fh_Kpong.txt"),
            self.n_years, self.n_days_one_year
            ) # historical fixed head Kpong (ft) 1984-2012
        
            
    def set_log(self, log_objectives):
        if log_objectives:
            self.log_objectives = True
        else:
            self.log_objectives = False

    def get_log(self):
        return self.blevel_Ak, self.rirri, self.rflood, self.renv           #***add rarea for inundated area ?
                
    def apply_rbf_policy(self, rbf_input):

        # normalize inputs (divide input by maximum)
        formatted_input = rbf_input / self.input_max                            #from line 121. still not sure what it is

        # apply rbf
        normalized_output = self.rbf.apply_rbfs(formatted_input)

        # scale back normalized output (multiply output by maximum)
        scaled_output = normalized_output * self.output_max

        # uu = []
        # for i in range(0, self.):
        #     uu.append(u[i] * self.output_max[i])
        return scaled_output

    def evaluate_historic(self, var, opt_met=1):
        return self.simulate(var, self.evap_Ak, self.inflow_Ak,
                             self.tailwater_Ak, self.fh_Kpong, opt_met)
    
    
    def evaluate_mc(self, var, opt_met=1):
        obj, Jhyd_a, Jhyd_d, Jirri, Jflood, Jenv = [], [], [], [], [], []       #add inundation function?
        # MC simulations
        n_samples = 2 #historic and stochastic?
        for i in range(0, n_samples):
            Jhydropower_a, Jhydropower_d, Jirrigation, Jfloodcontrol, \
            Jenvironment = self.simulate(
                var,
                self.evap_Ak,
                self.inflow_Ak,
                self.tailwater_Ak, 
                self.fh_Kpong,
                opt_met,
            )
            Jhyd_a.append(Jhydropower_a)
            Jhyd_d.append(Jhydropower_d)
            Jirri.append(Jirrigation)
            Jflood.append(Jfloodcontrol)
            Jenv.append(Jenvironment)

        # objectives aggregation (minimax)
        obj.insert(0, np.percentile(Jhyd_a, 99))
        obj.insert(1, np.percentile(Jhyd_d, 99))
        obj.insert(2, np.percentile(Jirri, 99))
        obj.insert(3, np.percentile(Jflood, 99))
        obj.insert(4, np.percentile(Jenv, 99))
        return obj
    
    #convert storage at current timestep to level and then level to surface area
    def storage_to_level(self, s):
        # s : storage
        # lake : which lake it is at            #only Akosombo 
        # gets triggered decision step * time horizon
        s_ = utils.cubicFeetToAcreFeet(s)
        h = utils.interpolate_linear(self.lsv_rel[2], self.lsv_rel[0], s_)
        return h
    
    def level_to_storage(self, h):
        s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[2], h)        
        return utils.acreFeetToCubicFeet(s)

    def level_to_surface(self, h):
        s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[1], h)
        return utils.acreToSquaredFeet(s)
    
    """comes into play in Sasquehana because Muddy Run is a parallel system. For Volta, self.actual_release aplies
    def akosombo_turb(self, level_Ak):
        # Determines the turbine release volumes at Akosombo in a day 
        # Release from Kpong equal to the release from Akosombo
        QT = 56166  # cfs max turbine capacity at Akosombo (and Kpong)

        # active storage = current storage - deadStorage
        qM = (self.level_to_storage(level_Ak, 0) -
              self.level_to_storage(240.0, 0))/3600     #divide by 3600 to convert from volume (cubic ft) to flowrate (cfs)
        
        qt = min(QT, qM)    #flow through turbines 
       
        if level_Ak < 240:
            qt = 0.0    #no flow through turbines if water level is below dead storage
        return qt  # flow through turbine
"""

    def actual_release(self, uu, level_Ak, day_of_year): #including flow through spillways
        # Check if flow exceeds the spillway capacity?              *** but what happens in that scenario?
        #uu = prescribed release policy        
        Tcap = 56166  # total turbine capacity at Akosombo(cfs)
        maxSpill = 730800 # total spillway capacity (cfs)
        
        # minimum discharge values for irrigation, downstream and flood releases
        qm_I = 0.0
        qm_D = 0.0
        qm_F = 0.0

        # maximum discharge values. The max discharge can be as much as the
        # demand in that area
        qM_I = self.annual_irri[day_of_year] #  irrigation demand is an annual demand...*** 
        qM_D = Tcap
        qM_F = maxSpill  

        # reservoir release constraints
        if level_Ak <= self.min_level_Akosombo:
            qM_I = 0.0
        else:
            qM_I = self.annual_irri[day_of_year]
        
        if level_Ak > self.flood_warn_level:  # spillways activated
            qM_D = Tcap #turbine capacity
            qM_F = (
                    utils.interpolate_linear(self.spillways[0],
                                             self.spillways[1],
                                             level_Ak)
            )  # spillways
            qm_F = (
                    utils.interpolate_linear(self.spillways[0],
                                             self.spillways[1],
                                             level_Ak) 
                    )

        # different from flood model            #what does this mean***?

        # actual release
        rr = []
        rr.append(min(qM_I, max(qm_I, uu[0])))
        rr.append(min(qM_D, max(qm_D, uu[1])))
        rr.append(min(qM_F, max(qm_F, uu[2])))
        return rr

    @staticmethod
    @njit
    def g_hydAk(r, h, day_of_year, GG, gammaH20, tailwater, turbines):
     # hydropower @ Akosombo =f(release through turbines, water level/headwater level, day of year, *hour0 
                             #(if considering sub daily timestep), gravitational acc,
                             #water density, tailwater level, flow through turbines)  
             
        cubicFeetToCubicMeters = 0.0283  # 1 cf = 0.0283 m3
        feetToMeters = 0.3048  # 1 ft = 0.3048 m
        Nturb = 6 #number of turbines at Akosombo
        g_hyd = [] 
        pp = [] #power generated in GWh
        for i in range(0, len(r)):
            deltaH = h[i] - tailwater[i], #water level (h) - tailwater level (self.tailwater?) on the ith day = net hydraulic head  #is this correct?****
            q_split = r[i]      #for when some turbines are shut
            for j in range(0, Nturb):
                if q_split < turbines[1][j]: #not clear ...if the turbine flow is less than what?***
                    qturb = 0.0
                elif q_split > turbines[0][j]:
                    qturb = turbines[0][j]
                else:
                    qturb = q_split
                q_split = q_split - qturb
                p = (
                        0.93 #efficiency of Akosombo plant
                        * GG #acceleration to gravity
                        * gammaH20 #density of water
                        * (cubicFeetToCubicMeters * qturb) #flow through turbines
                        * (feetToMeters * deltaH) # net hydraulic head
                        * 3600     #power generated Wh
                        / (24 * 1000000) # conversion from Wh to GWh/day
                )  # daily energy prouction
                pp.append(p)
            g_hyd.append(np.sum(np.asarray(pp)))
            pp.clear()
        Gp = np.sum(np.asarray(g_hyd))
        return Gp

    @staticmethod
    @njit
    def g_hydKp(r, fixedhead, day_of_year, GG, gammaH20, turbines_Kp,
                   ):
     # hydropower @ Kpong =f(flow release, fixed head, day of year, *hour0 
                             #(if considering sub daily timestep), gravitational acc,
                             #water density, flow through turbines)  
     
        cubicFeetToCubicMeters = 0.0283  # 1 cf = 0.0283 m3
        feetToMeters = 0.3048  # 1 ft = 0.3048 m
        Nturb = 4 #number of turbines at Kpong
        g_hyd_Kp = []
        pp_K = [] #power generated in GWh
        for i in range(0, len(r)):
            deltaH =  fixedhead[i], #fixed head             *** is this correct?
            q_split = r[i]      #for when some turbines are shut
            for j in range(0, Nturb):
                if q_split < turbines_Kp[1][j]:
                    qturb = 0.0
                elif q_split > turbines_Kp[0][j]:
                    qturb = turbines_Kp[0][j]
                else:
                    qturb = q_split
                q_split = q_split - qturb
                p = (
                        0.93 #efficiency of Akosombo plant
                        * GG #acceleration to gravity
                        * gammaH20 #density of water
                        * (cubicFeetToCubicMeters * qturb) #flow through turbines
                        * (feetToMeters * deltaH) # net hydraulic head
                        * 3600     #power generated Wh
                        / (24 * 1000000) # conversion from Wh to GWh/day
                )  # daily energy prouction
                pp_K.append(p)
            g_hyd_Kp.append(np.sum(np.asarray(pp_K)))
            pp_K.clear()
        Gp_Kp = np.sum(np.asarray(g_hyd_Kp))
        return Gp_Kp

    def res_transition_h(self, s0, uu, n_sim, ev, n_sim_kp,
                          day_of_year):
        #s0-storage_Akosombo, uu- prescribed release policy, n_sim-??*****
        # ev- evaporation_Ak, #n_sim_kp-??, day_of_year
        HH = self.hours_between_decisions  
        sim_step = 3600  # seconds per hour
        leak = 0  # cfs loss to groundwater and other leaks. (like in WEAP model = 0)

        # Storages and levels of Akosombo and Kpong
        shape = (HH+1, ) #?? what is shape?

        storage_Ak = np.empty(shape)
        level_Ak = np.empty(shape)
        
        # Actual releases (Irrigation, Dowstream, flood, hydropower)
        shape = (HH,) 
        release_I = np.empty(shape) #irrigation
        release_D = np.empty(shape) # downstream relaease including e-flows
        release_F = np.empty(shape) #flood releases when water levels are high
        q_rel = np.empty(shape)  # release through turbines 
        
        # initial conditions
        storage_Ak[0] = s0

        for i in range(0, HH):
            # compute level @ Akosombo
            level_Ak[i] = self.storage_to_level(storage_Ak[i], 1)
            # Akosombo operation
            #q_rel[i] = self.akosombo_turb(level_Ak[i]) #I don't think I need this as

            # Compute actual release
            rr = self.actual_release(uu, level_Ak[i], day_of_year)
            release_I[i] = rr[0]
            release_D[i] = rr[1]
            release_F[i] = rr[2]

            # Q: Why is this being added?
            # FIXME: actual release as numpy array then simple sum over slice
            # FIXME: into rr
            #WS = release_I[i]

            # Compute surface level and evaporation loss
            surface_Ak = self.level_to_surface(level_Ak[i], 1)
            evaporation_losses_Ak = utils.inchesToFeet(ev) * surface_Ak / 86400 
                                     # cfs _ daily ET
            
            # System Transition
            storage_Ak[i + 1] = storage_Ak[i] + sim_step * (        #multiplied by sim_step (3600) to convert from flow (cfs) to volume (cubic feet)
                    n_sim - evaporation_losses_Ak  -release_I[i] 
                    -release_D[i]- release_F[i] - leak #- q_rel[i]
            ) 

        sto_ak = storage_Ak[HH]
        rel_i = utils.computeMean(release_I)
        rel_d = utils.computeMean(release_D)
        rel_f = utils.computeMean(release_F)

        level_Ak = np.asarray(level_Ak)
        hp = VoltaModel.g_hydAk(
            np.asarray(release_D),
            level_Ak,
            day_of_year,
            self.GG,
            self.gammaH20,
            self.tailwater,
            self.turbines,
        )
        
        hp_kp = VoltaModel.g_hydKp(
            np.asarray(release_D),
            self.fh_Kpong,
            day_of_year,
            self.GG,
            self.gammaH20,
            self.turbines_Kp,
        )
        
        return sto_ak, rel_i, rel_d, rel_f, hp[0], hp_kp[0]

    # Minimization function of number of flood events
    def g_storagereliability(self, h, h_target):        #flood control objective which limits all floods
        c = 0
        #Nw = 0

        # FIXME this probably can be fully vectorized
        # the modulus is not neeede dif h_target is just
        # expanded to match the length of h
        for i, h_i in np.ndenumerate(h):            #enumerates each element in an N-dimensional array
            tt = i[0] % self.n_days_one_year        #this is to get the day in the year. Dividing the ith timestep in the simulation by 365 and taking the reminder gives the day in the year
            if h_i < h_target[tt]:                  #if the water level on the day is less than the target level for that day
                c = c + 1
            # if h_target[tt] > 0:
            #     Nw += 1

        G = 1 - c / np.sum(h_target > 0)
        return G

    # Minimization function of area inundated during floods ****need help to fix this
    def g_inundated_area(self, q, qThreshold):        
        A = 0   
        #*** when there is flow through spillways  recall(qM_F = maxSpill)
        #add q_F and q_D to get total flow downstream
        for i, q_i in np.ndenumerate(q):
            #divide by (3600) to convert from volume to flow
            #cfs_to_cumecs =   0.028316847  
            if q_i <= 2300:
                A == 0
            elif 2300 < q_i <= 3000:
                A == (0.2229 * q_i) - 512.57
            elif 3000< q_i <= 10000:
                A == ((2 * 10**-6)* (q_i**2)) - (0.0098*q_i) + 173.9
            elif q_i > 10000:
                A == ((3 * 10**-5)* (q_i**2)) - (0.593*q_i) + 3202.6
            else:
                A == 818.3

            return A #inundated are in km2

    #Minimization function of the difference between q1 and qTarget with larger differences penalized more (hence the squaring)
    def g_shortage_index(self, q1, qTarget):        #for e-flows *** is this right? by using conditions, does the specified value for self.clam_eflows matter? 
        for i, q1_i in np.ndenumerate(q1):          #enumerates each element in an N-dimensional array
            tt = i[0] % self.n_days_one_year        ## dividing the ith timestep in the simulation by 365 and taking the reminder gives the day in the year
            while tt <= 90 or tt >= 305:
                if q1_i >= 300:
                    qTarget[tt] == 300
                elif q1_i <= 50:
                    qTarget[tt] == 50
                else:
                    qTarget[tt] = q1
    
        delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q1) / self.n_days_one_year)) #This is just a transformation to get qTarget in the same dimension/length as q1
        maxarr = abs((qTarget * delta) - (q1 * delta))
        gg = maxarr / (qTarget * delta)
        g = np.mean(np.square(gg))
        return g
    
    #Minimization function of the difference between the q1 and qTarget (for irrigation)
    def g_vol_rel(self, q1, qTarget):       #q1 = flow on ith day;      qTarget = target flow/water demand
        delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q1) / self.n_days_one_year))         #if I don't divide by 365, is q1 then an annual flow?** 
        #This is just a transformation to get qTarget in the same dimension/length as q1       
        g = (q1 * delta) / (qTarget * delta)    #multiplied by 24*3600 to convert from flow to volume
        G = utils.computeMean(g)
        return G
    
    
    #Minimization function of the difference between the p1 and pTarget (for annual hydropower)
    def g_hydro_rel(self, p1, pTarget):       
        pTarget = np.tile(pTarget, int(len(p1)))      
        g = (p1 - pTarget)    
        G = np.mean(np.square(g))
        return G

    def simulate(self, input_variable_list_var, inflow_Ak_n_sim,
                 fixedhead_Kpong_n_kp, evap_Ak_e_ak,  opt_met):
        # Initializing daily variables
        # storages and levels

        shape = (self.time_horizon_H + 1, )
        storage_ak = np.empty(shape)
        level_ak = np.empty(shape)

        # Akosombo actual releases
        shape = (self.time_horizon_H,)
        release_i = np.empty(shape) #irrigation
        release_d = np.empty(shape) #downstream release (hydro, e-flows)
        release_f = np.empty(shape) #flood
        
        # hydropower production/
        hydropowerProduction_Ak = []  # energy production at Akosombo
        hydropowerProduction_Kp = []  # energy production at Kpong

        # release decision variables ( irrigation ) only
        # Downstream in Baseline
        self.rbf.set_decision_vars(np.asarray(input_variable_list_var))

        # initial condition
        level_ak[0] = self.init_level
        storage_ak[0] = self.level_to_storage(level_ak[0], 1)

        # identification of the periodicity (365 x fdays)
        decision_steps_per_year = self.n_days_in_year * self.decisions_per_day
        year = 0

        # run simulation
        for t in range(self.time_horizon_H):
            day_of_year = t % self.n_days_in_year
            if day_of_year % self.n_days_in_year == 0 and t != 0:
                year = year + 1

            # subdaily variables
            shape = (self.decisions_per_day + 1,)
            daily_storage_ak = np.empty(shape)
            daily_level_ak = np.empty(shape)

            shape = (self.decisions_per_day,)
            daily_release_i = np.empty(shape)
            daily_release_d = np.empty(shape)
            daily_release_f = np.empty(shape)

            # initialization of sub-daily cycle             ***is this necessary for Volta where timestep is daily?
            daily_level_ak[0] = level_ak[t]  # level_ak[day_of_year] <<< in flood
            daily_storage_ak[0] = storage_ak[t]

            # sub-daily cycle
            for j in range(self.decisions_per_day):
                decision_step = t * self.decisions_per_day + j

                # decision step i in a year
                jj = decision_step % decision_steps_per_year

                # compute decision
                if opt_met == 0:  # fixed release
                    # FIXME will crash because uu is empty list
                    uu.append(uu[0])
                elif opt_met == 1:  # RBF-PSO
                    rbf_input = np.asarray([jj, daily_level_ak[j]])
                    uu = self.apply_rbf_policy(rbf_input)

                # system transition
                ss_rr_hp = self.res_transition_h(
                    daily_storage_ak[j],
                    uu,
                    inflow_Ak_n_sim[year][day_of_year],
                    evap_Ak_e_ak[year][day_of_year],
                    fixedhead_Kpong_n_kp[year][day_of_year],
                    day_of_year,
                    j,
                )

                daily_storage_ak[j+1] = ss_rr_hp[0]
                daily_level_ak[j+1] = self.storage_to_level(daily_storage_ak[j+1], 1)

                daily_release_i[j] = ss_rr_hp[1]
                daily_release_d[j] = ss_rr_hp[2]
                daily_release_f[j] = ss_rr_hp[3]

                # Hydropower production
                hydropowerProduction_Ak.append(
                    ss_rr_hp[4])  # daily energy production (MWh/day) at Akosombo
                hydropowerProduction_Kp.append(
                    ss_rr_hp[5])  # daily energy production (MWh/day) at Kpong

            # daily values
            level_ak[day_of_year + 1] = daily_level_ak[self.decisions_per_day]
            storage_ak[t + 1] = daily_storage_ak[self.decisions_per_day]
            release_i[day_of_year] = np.mean(daily_release_i)
            release_d[day_of_year] = np.mean(daily_release_d)
            release_f[day_of_year] = np.mean(daily_release_f)

        # log level / release
        if self.log_objectives:
            self.blevel_Ak.append(level_ak)
            self.rirri.append(release_i)
            self.renv.append(release_d) 
            self.rflood.append(release_f)

        # compute objectives
        j_hyd_a = sum(hydropowerProduction_Ak, hydropowerProduction_Kp) / self.n_years   # GWh/year #maximization 
        #j_hyd_a = self.g_hydro_rel((sum(hydropowerProduction_Ak, hydropowerProduction_Kp)),self.annual_power) #minimization of deviation from target of 4415GWh
        
        j_hyd_d = sum(hydropowerProduction_Ak, hydropowerProduction_Kp) / self.time_horizon_H #GWh/day  (self.time_horizon_H= self.n_days_in_year * self.n_years)
        #2nd Maximization of daily hydropower at 90 per cent reliability over the hydrological ensemble
        
        j_irri = self.g_vol_rel(release_i, self.annual_irri)
        j_env = self.g_shortage_index(release_d, self.clam_eflows)
        j_fldcntrl = self.g_storagereliability(storage_ak, self.flood_protection)#
        j_inun = self.inundated_area(release_f, self.inundated_area)

        return j_hyd_a, j_hyd_d, j_irri, j_env, j_fldcntrl, j_inun
            



        
