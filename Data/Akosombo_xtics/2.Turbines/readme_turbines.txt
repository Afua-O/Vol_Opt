Max and min capacity and efficiency of each turbine. There are 6 turnines in total @ Akosombo

Max capacity of  all 6 turbines: 56616 cfs (1,600m3/s) (Output=1038MW)
Unit max capacity = 9436 cfs
Min capacity (all 6): 5050 cfs
Unit min capacity = 841.67 cfs
Efficiency: 90% up to 2006 and 93% since (after retrofit)


Additional info: The maximum hydraulic outflow (spillway + Turbines) is 22,297m3/s. 

The minimum turbine flow was calculated using:
- minimum daily power requirement (for system stability) of 6GWh/day (=0.25GW =6GWh/24h)
- minimum water level of Akosombo dam = 240 ft
- normal tailwater level @ Akosombo  =48.5 ft

HP (GW) = efficency (0.93) * gravitational acc * water density * net hydraulic head * turbine flow * 10^-9


self.turbines = utils.loadMatrix(create_path("./Data/Akosombo_xtics/2.Turbines/turbines_Ak.txt"), 3, 6)  
# Max-min capacity (cfs) - efficiency of Akosombo plant turbines
