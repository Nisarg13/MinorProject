import time
import mujoco.viewer


def simulate_with_viewer(model, data, viewer, max_wall_time=30):
    start_time = time.time()

    while viewer.is_running() and time.time() - start_time < max_wall_time:
        step_start_time = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def main():
    model_path = 'mujoco_menagerie/wonik_allegro/scene_left.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        simulate_with_viewer(model, data, viewer)


if __name__ == "__main__":
    main()
