import time
import mujoco.viewer
import numpy as np
import csv

ACTUATOR_LIST = ['ffa0', 'ffa1', 'ffa2', 'ffa3', 'mfa0', 'mfa1', 'mfa2', 'mfa3', 'rfa0', 'rfa1', 'rfa2', 'rfa3', 'tha0',
                 'tha1', 'tha2', 'tha3']

def generate_dynamic_control_signal(time_elapsed, frequency=0.5, amplitude=0.2, offset=0.0):
    return amplitude * np.sin(2 * np.pi * frequency * time_elapsed) + offset

def collect_training_data(time_history, control_signal_history, data):
    time_history.append(data.time)
    for name, value in zip(ACTUATOR_LIST, data.ctrl):
        control_signal_history[name].append(value)

def save_training_data_to_csv(file_path, time_history, control_signal_history):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Time'] + ACTUATOR_LIST
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for time_point in range(len(time_history)):
            row_data = {'Time': time_history[time_point]}
            for name in ACTUATOR_LIST:
                row_data[name] = control_signal_history[name][time_point]
            writer.writerow(row_data)

def simulate_with_viewer(model, data, viewer, max_wall_time=10, save_interval=5):
    start_time = time.time()
    save_start_time = start_time
    run_time = 0

    time_history = []
    control_signal_history = {name: [] for name in ACTUATOR_LIST}

    while viewer.is_running() and run_time < max_wall_time:
        step_start_time = time.time()

        dynamic_control_signal = generate_dynamic_control_signal(run_time)
        data.ctrl[:] = dynamic_control_signal

        collect_training_data(time_history, control_signal_history, data)

        if time.time() - save_start_time > save_interval:
            save_start_time = time.time()
            save_training_data_to_csv('training_data.csv', time_history, control_signal_history)

        mujoco.mj_step(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        run_time = time.time() - start_time

    save_training_data_to_csv('training_data.csv', time_history, control_signal_history)

def main():
    model_path = 'mujoco_menagerie/wonik_allegro/scene_right.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        simulate_with_viewer(model, data, viewer)

if __name__ == "__main__":
    main()
