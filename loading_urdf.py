import time
import mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animation


model = mujoco.MjModel.from_xml_path("./ARM_Main/urdf/ARM_Main.xml")
renderer = mujoco.Renderer(model, 480, 480)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data, "closeup")
plt.imshow(renderer.render())
plt.show()

# setup
n_seconds = 6
framerate = 30  # Hz
n_frames = int(n_seconds * framerate)
frames = []


# set initial state
mujoco.mj_resetData(model, data)
data.joint('root').qvel = 10


# simulate and record frames
frame = 0
sim_time = 0
render_time = 0
n_steps = 0
for i in range(n_frames):
    while data.time * framerate < i:
        tic = time.time()
        mujoco.mj_step(model, data)
        sim_time += time.time() - tic
        n_steps += 1
    tic = time.time()
    renderer.update_scene(data, "closeup")
    frame = renderer.render()
    render_time += time.time() - tic
    frames.append(frame)


def animate(frame):
    plt.clf()
    plt.imshow(frames[frame])
    plt.axis('off')


fig = plt.figure()
ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=17)
plt.show()
