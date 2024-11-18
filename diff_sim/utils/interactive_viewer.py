import mujoco
from mujoco import viewer, mj_step

def interactive_viewer(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[0], data.qpos[7] = 0.1, 0.175
    mj_step(model, data)
    with mujoco.viewer.launch(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", help="xml file to render", default="diff_sim/xmls/snake_mass.xml")
    interactive_viewer(parser.parse_args().xml)