import matplotlib.pyplot as plt

import numpy as np

from pyworld3 import World3, world3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

def run_world3_simulation(year_min, year_max, dt=1, prev_run_data=None, ordinary_run=True, k_index=1):
    
    prev_run_prop = prev_run_data["world_props"] if prev_run_data else None

    world3 = World3(
            year_max=year_max,
            year_min=year_min, 
            dt=dt,
            prev_world_prop=prev_run_prop,
            ordinary_run=ordinary_run
        )
    
    if prev_run_data:
        world3.set_world3_control(prev_run_data['control_signals'])
        world3.init_world3_constants()
        world3.init_world3_variables(prev_run_data["init_vars"])
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions(prev_delay=prev_run_data["delay_funcs"])
    else:
        world3.set_world3_control()
        world3.init_world3_constants()
        world3.init_world3_variables()
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions()

    world3.run_world3(fast=False, k_index=k_index)
    state = world3.get_state()
    return state, world3

def update_control(var, val, prev_control):
    for var in var:
        prev_control[var + '_control'] *= val
    return prev_control

def main():
    # Run the first simulation
    prev_data, world3 = run_world3_simulation(year_min=1900, year_max=2000)


    for year in range(2005, 2200, 5):
        # Run the second simulation with initial conditions derived from the first simulation
        prev_data, world3_current = run_world3_simulation(year_min=year-5, year_max=year, prev_run_data=prev_data, ordinary_run=False, k_index=prev_data["world_props"]["k"])

        control_variables = ['ppgf']
        # prev_data['control_signals'] = update_control(control_variables, 0.9, prev_data['control_signals'])

    variables = [world3_current.hsapc, world3_current.p1, world3_current.ehspc]
    labels = ["HSAPC", "P1", "EHSPC"]
   
    # Plot the combined results
    plot_world_variables(
        world3_current.time,
        variables,
        labels,
        [[0, 1.5*max(world3_current.hsapc)], [0, 4e9], [0, 80]],
        figsize=(10, 7),
        title="World3 Simulation from 1900 to 2200, paused at 2000"
    )
    # Initialize a position for the first annotation
    x_pos = 0.05  # Adjust as needed
    y_pos = 0.95  # Start from the top, adjust as needed
    vertical_offset = 0.05  # Adjust the space between lines

    # Use plt.gcf() to get the current figure and then get the current axes with gca()
    ax = plt.gcf().gca()

    for var, label in zip(variables, labels):
        max_value = np.max(var)
        # Place text annotation within the plot, using figure's coordinate system
        ax.text(x_pos, y_pos, f'{label} Max: {max_value:.2f}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')
        y_pos -= vertical_offset  # Move up for the next line

    plt.show()

if __name__ == "__main__":
    main()

