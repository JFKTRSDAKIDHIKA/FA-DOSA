import json
import matplotlib.pyplot as plt
import numpy as np

# Load the optimization log
with open('optimization_log.jsonl', 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

# Extract data for plotting
phases = [d['phase'] for d in data]
steps = [d['outer_step'] * 1000 + d['inner_step'] for d in data]  # Combined step for x-axis
total_loss = [d['loss_total'] for d in data]
edp_loss = [d['loss_components']['edp'] for d in data]
area_loss = [d['loss_components']['area'] for d in data]
latency = [d['performance_metrics']['latency_sec'] for d in data]
energy = [d['performance_metrics']['energy_pj'] for d in data]
area = [d['performance_metrics']['area_mm2'] for d in data]
num_pes = [d['hardware_params']['num_pes'] for d in data]
l1_size = [d['hardware_params']['l1_size_kb'] for d in data]
l2_size = [d['hardware_params']['l2_size_kb'] for d in data]
tau = [d['gumbel_tau'] for d in data]

# Create a figure with multiple subplots
plt.figure(figsize=(15, 20))

# Plot 1: Loss Components
plt.subplot(4, 1, 1)
plt.plot(steps, total_loss, 'k-', label='Total Loss')
plt.plot(steps, edp_loss, 'r-', label='EDP Loss')
plt.plot(steps, area_loss, 'b-', label='Area Loss')

# Add vertical lines to separate phases
phase_a_indices = [i for i, phase in enumerate(phases) if phase == 'A: Mapping']
phase_b_indices = [i for i, phase in enumerate(phases) if phase == 'B: Hardware']

# Find the boundaries between phases
phase_boundaries = []
for i in range(1, len(phases)):
    if phases[i] != phases[i-1]:
        phase_boundaries.append(steps[i])

for boundary in phase_boundaries:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

plt.title('Loss Components Over Time')
plt.xlabel('Optimization Step')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True)

# Plot 2: Performance Metrics
plt.subplot(4, 1, 2)
plt.plot(steps, latency, 'g-', label='Latency (s)')
plt.plot(steps, [e / 1e14 for e in energy], 'm-', label='Energy (1e14 pJ)')
plt.plot(steps, area, 'c-', label='Area (mm²)')

for boundary in phase_boundaries:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

plt.title('Performance Metrics Over Time')
plt.xlabel('Optimization Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot 3: Hardware Parameters
plt.subplot(4, 1, 3)
plt.plot(steps, num_pes, 'b-', label='Number of PEs')
plt.plot(steps, l1_size, 'r-', label='L1 Size (KB)')
plt.plot(steps, l2_size, 'g-', label='L2 Size (KB)')

for boundary in phase_boundaries:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

plt.title('Hardware Parameters Over Time')
plt.xlabel('Optimization Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot 4: Gumbel-Softmax Temperature
plt.subplot(4, 1, 4)
plt.plot(steps, tau, 'r-', label='Gumbel-Softmax Temperature (tau)')

for boundary in phase_boundaries:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

plt.title('Gumbel-Softmax Temperature Over Time')
plt.xlabel('Optimization Step')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('optimization_visualization.png', dpi=300)
plt.close()

print("Visualization saved to 'optimization_visualization.png'")