import sys
import os
import numpy as np
import pandas as pd

# Add project root to path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import MARLEdgeInferenceEnv

def test_transmission_rates_statistics(num_timesteps=100):
    """
    Initializes the environment and runs multiple independent "timesteps"
    by re-randomizing positions/channels to gather statistics on transmission rates.
    """
    print(f"--- Initializing MARLEdgeInferenceEnv for statistical test over {num_timesteps} timesteps ---")
    
    # Use a specific seed for the environment's RNG for reproducible user/server properties
    env = MARLEdgeInferenceEnv(config={'env': {'seed': 42}})
    print(f"✅ Environment initialized with {env.num_users} users and {env.num_edges} servers.")
    print("-" * 70)

    all_results = []
    
    for t in range(num_timesteps):
        # Re-initialize communication params to get new positions and channel gains (shadowing)
        env._init_communication_params()

        # Simple association: cyclically assign users to servers
        users_per_server = int(np.ceil(env.num_users / env.num_edges))

        for j, server in enumerate(env.servers):
            start_user_idx = j * users_per_server
            end_user_idx = min((j + 1) * users_per_server, env.num_users)
            associated_user_indices = list(range(start_user_idx, end_user_idx))

            if not associated_user_indices:
                continue

            total_bw_hz = server.bandwidth_MHz * 1e6
            bw_alloc_hz = total_bw_hz / len(associated_user_indices)

            for k in associated_user_indices:
                user = env.users[k]
                rate_down_bps, rate_up_bps = env._calculate_transmission_rates(
                    server_id=j, user_id=k, server=server, user=user, bw_alloc_hz=bw_alloc_hz
                )
                all_results.append({
                    "Timestep": t,
                    "UserID": k,
                    "ServerID": j,
                    "Distance (m)": env.distances_m[j, k],
                    "Channel Gain": env.channel_gains[j, k],
                    "Alloc BW (MHz)": bw_alloc_hz / 1e6,
                    "Downlink (Mbps)": rate_down_bps / 1e6,
                    "Uplink (Mbps)": rate_up_bps / 1e6,
                })
        
        if (t + 1) % 10 == 0:
            print(f"  ... completed timestep {t + 1}/{num_timesteps}")

    # --- Analyze and display statistics ---
    if not all_results:
        print("No results to display.")
        return

    df = pd.DataFrame(all_results)
    
    print("\n--- Statistical Summary of Transmission Rates ---")
    
    # Describe provides a good summary (mean, std, min, max, quartiles)
    summary = df[['Downlink (Mbps)', 'Uplink (Mbps)']].describe().transpose()
    print(summary.to_string())
    
    print("\n--- Extreme Cases Found ---")
    
    # Find and display the best and worst cases for downlink
    best_downlink = df.loc[df['Downlink (Mbps)'].idxmax()]
    worst_downlink = df.loc[df['Downlink (Mbps)'].idxmin()]
    
    # Find and display the best and worst cases for uplink
    best_uplink = df.loc[df['Uplink (Mbps)'].idxmax()]
    worst_uplink = df.loc[df['Uplink (Mbps)'].idxmin()]
    
    print("\n[Best Downlink Case]")
    print(best_downlink)
    
    print("\n[Worst Downlink Case]")
    print(worst_downlink)
    
    print("\n[Best Uplink Case]")
    print(best_uplink)
    
    print("\n[Worst Uplink Case]")
    print(worst_uplink)

    print("-" * 70)
    print("Statistical test finished.")

if __name__ == "__main__":
    # You can change the number of timesteps to run here
    test_transmission_rates_statistics(num_timesteps=100)
