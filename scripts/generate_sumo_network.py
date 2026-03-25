import os
from pathlib import Path
import subprocess

def create_sumo_network(output_dir: str):
    """
    Generate a 2x2 grid network compatible with the sumo_env simulator.
    Assumes SUMO is installed and bin/ is in the PATH (netgenerate, jtrrouter etc).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    net_file = out_path / "grid.net.xml"
    sumocfg_file = out_path / "sim.sumocfg"
    route_file = out_path / "routes.rou.xml"

    # 1. Generate 2x2 Network using netgenerate
    print("Generating SUMO 2x2 Network...")
    cmd = [
        "netgenerate", 
        "--grid", 
        "--grid.x-number", "2", 
        "--grid.y-number", "2", 
        "--grid.x-length", "400", 
        "--grid.y-length", "400", 
        "--grid.attach-length", "400",
        "--default-junction-type", "traffic_light",
        "--output-file", str(net_file)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("\nERROR: 'netgenerate' not found. Please install SUMO and add its bin/ to your system PATH.")
        print("Download SUMO: https://sumo.dlr.de/docs/Downloads.html\n")
        return False
        
    # 2. Make dummy routes file for valid edges
    route_xml = """<routes>
    <vType id="car" length="5.0" maxSpeed="15.0" accel="2.6" decel="4.5" />
    
    <!-- Real Routes for A0, A1, B0, B1 -->
    <route id="route_ns_0" edges="bottom0A0 A0A1 A1top0" />
    <route id="route_ew_0" edges="left0A0 A0B0 B0right0" />
    
    <route id="route_ns_1" edges="bottom0A0 A0A1 A1top0" />
    <route id="route_ew_1" edges="left1A1 A1B1 B1right1" />

    <route id="route_ns_2" edges="bottom1B0 B0B1 B1top1" />
    <route id="route_ew_2" edges="left0A0 A0B0 B0right0" />
    
    <route id="route_ns_3" edges="bottom1B0 B0B1 B1top1" />
    <route id="route_ew_3" edges="left1A1 A1B1 B1right1" />
</routes>"""
    route_file.write_text(route_xml)

    # 3. Make sumocfg
    cfg_xml = f"""<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
    </time>
</configuration>"""
    sumocfg_file.write_text(cfg_xml)
    
    print(f"SUMO config skeleton generated at: {sumocfg_file}")
    return True

if __name__ == "__main__":
    create_sumo_network("data/sumo")
