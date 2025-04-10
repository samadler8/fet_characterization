#%%
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.agilent_34411a import Agilent34411A


ch_s = 2
ch_gate = 3

srs_s = SIM928('GPIB0::2::INSTR', ch_s)
srs_gate = SIM928('GPIB0::2::INSTR', ch_gate)
srs_dmm = SIM970('GPIB0::2::INSTR', 7)
multi1 = Agilent34411A('GPIB0::22::INSTR')
multi2 = Agilent34411A('GPIB0::21::INSTR')

rbias = 4.991e3
voltage_settle_time = 0.1

fet_pn = 'J113FS-ND'
test_fet_num = 1
fet_name = f'{fet_pn}_{test_fet_num}'

#%%
def measure_fet_fixed_vgs(min_vgs, max_vgs, max_vbias, num_vgs, num_vbias):
    vgss = np.linspace(min_vgs, max_vgs, num_vgs)
    vbiases = np.linspace(0, max_vbias, num_vbias)


    N = 2
    data_list = []

    srs_gate.set_voltage(0)
    srs_gate.set_output(output=True)

    srs_s.set_voltage(0)
    srs_s.set_output(output=True)
    for vgs in tqdm(vgss):

        srs_gate.set_voltage(vgs)

        for vbias in vbiases:

            srs_s.set_voltage(vbias)

            time.sleep(voltage_settle_time)

            # v_1 = sum(srs_dmm.read_voltage(1) for _ in range(N)) / N
            # v_2 = sum(srs_dmm.read_voltage(2) for _ in range(N)) / N

            v_1 = sum(multi1.read_voltage() for _ in range(N)) / N
            v_2 = sum(multi2.read_voltage() for _ in range(N)) / N

            data = dict(
                vbias = v_1,
                vds = v_2,
                vgs = vgs,
                ids = (v_1 - v_2)/rbias,
                Rbias = rbias,
            )
            data_list.append(data)

    srs_gate.set_voltage(0)
    srs_gate.set_output(output=False)

    srs_s.set_voltage(0)
    srs_s.set_output(output=False)

    df = pd.DataFrame(data_list)

    df['Rfet'] = df['vds']/df['ids']
    df['Gfet'] = 1/df['Rfet']

    return df



#%%
min_vgs = -1.1
max_vgs = -1.5
max_vbias = 5
num_vgs = 10
num_vbias = 100

df_vgs = measure_fet_fixed_vgs(min_vgs, max_vgs, max_vbias, num_vgs, num_vbias)
df_vgs.to_csv(f'fet_output_data_vgs_{fet_name}.csv', index=False)



# %%

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot I_D vs V_DS
for vgs, subdf in df_vgs.groupby('vgs'):
    axes[0].plot(subdf['vds'], subdf['ids'], label=f"Vgs = {vgs} V")

axes[0].set_ylabel('I_D (A)')
axes[0].set_title('FET Output Characteristics (I_D vs V_DS)')
axes[0].legend()
axes[0].grid(True)

# Plot Rfet vs V_DS (with cutoff at 1e6 Ohm)
for vgs, subdf in df_vgs.groupby('vgs'):
    subdf_cut = subdf.copy()
    subdf_cut = subdf_cut[subdf_cut['Rfet'] <= 1e6]  # Apply cutoff
    axes[1].plot(subdf_cut['vds'], subdf_cut['Rfet'], label=f"Vgs = {vgs} V")

axes[1].set_ylabel('R_fet (Ohms)')
axes[1].set_title('FET Resistance (Rfet vs V_DS)')
axes[1].legend()
axes[1].grid(True)

# Plot Gfet vs V_DS
for vgs, subdf in df_vgs.groupby('vgs'):
    axes[2].plot(subdf['vds'], subdf['Gfet'], label=f"Vgs = {vgs} V")

axes[2].set_xlabel('V_DS (V)')
axes[2].set_ylabel('G_fet (S)')
axes[2].set_title('FET Conductance (Gfet vs V_DS)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
fig.savefig(f'fet_output_characteristics_vds_{fet_name}.png', dpi=300)
plt.close()
print("Figure saved to 'fet_output_characteristics.png'")

#%%
def vds_binary_search(target_vds):
    N = 2
    tolerance = 0.05

    max_v = 20
    min_v = 0
    v_guess = 0
    meas_vds = 0
    difference = abs(meas_vds - target_vds)

    while difference > tolerance:
        if v_guess >= max_v - tolerance:
            print("ERROR: Can't set voltage high enough.")
            return None
        elif meas_vds < target_vds:
            min_v = v_guess
        else:
            max_v = v_guess 
        v_guess = (max_v + min_v) / 2

        srs_s.set_voltage(v_guess)
        time.sleep(voltage_settle_time)
        meas_vds = sum(multi2.read_voltage() for _ in range(N)) / N
        difference = abs(meas_vds - target_vds)

    return meas_vds




def measure_fet_fixed_vds(min_vgs, max_vgs, max_vds, num_vgs, num_vds):
    vgss = np.linspace(min_vgs, max_vgs, num_vgs)
    vdss = np.linspace(0, max_vds, num_vds)


    N = 2
    data_list = []

    srs_gate.set_voltage(0)
    srs_gate.set_output(output=True)

    srs_s.set_voltage(0)
    srs_s.set_output(output=True)
    for vgs in tqdm(vgss):

        srs_gate.set_voltage(vgs)

        for vds in vdss:

            v_2 = vds_binary_search(vds)
            
            if v_2 is None:
                break

            # time.sleep(voltage_settle_time)

            # v_1 = sum(srs_dmm.read_voltage(1) for _ in range(N)) / N
            # v_2 = sum(srs_dmm.read_voltage(2) for _ in range(N)) / N

            v_1 = sum(multi1.read_voltage() for _ in range(N)) / N

            data = dict(
                vbias = v_1,
                vds = v_2,
                vgs = vgs,
                ids = (v_1 - v_2)/rbias,
                Rbias = rbias,
            )
            data_list.append(data)

    srs_gate.set_voltage(0)
    srs_gate.set_output(output=False)

    srs_s.set_voltage(0)
    srs_s.set_output(output=False)

    df = pd.DataFrame(data_list)

    df['Rfet'] = df['vds']/df['ids']
    df['Gfet'] = 1/df['Rfet']

    return df


#%%
min_vgs = -1.1
max_vgs = -1.5
max_vds = 2
num_vgs = 100
num_vds = 5

df_vds = measure_fet_fixed_vds(min_vgs, max_vgs, max_vds, num_vgs, num_vds)
df_vds.to_csv(f'fet_output_data_vds_{fet_name}.csv', index=False)
#%%

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot I_D vs V_DS
for vds, subdf in df_vds.groupby('vds'):
    axes[0].plot(subdf['vgs'], subdf['ids'], label=f"Vds = {vds} V")

axes[0].set_ylabel('I_D (A)')
axes[0].set_title('FET Output Characteristics (I_D vs V_GS)')
axes[0].legend()
axes[0].grid(True)


# Plot Rfet vs V_DS (with cutoff at 1e6 Ohm)
for vgs, subdf in df_vgs.groupby('vds'):
    subdf_cut = subdf.copy()
    subdf_cut = subdf_cut[subdf_cut['Rfet'] <= 1e6]  # Apply cutoff
    axes[1].plot(subdf_cut['vgs'], subdf_cut['Rfet'], label=f"Vds = {vds} V")

axes[1].set_ylabel('R_fet (Ohms)')
axes[1].set_title('FET Resistance (Rfet vs V_GS)')
axes[1].legend()
axes[1].grid(True)

# Plot Gfet vs V_DS
for vds, subdf in df_vds.groupby('vds'):
    axes[2].plot(subdf['vgs'], subdf['Gfet'], label=f"Vds = {vds} V")

axes[2].set_xlabel('V_GS (V)')
axes[2].set_ylabel('G_fet (S)')
axes[2].set_title('FET Conductance (Gfet vs V_GS)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
fig.savefig(f'fet_output_characteristics_vgs_{fet_name}.png', dpi=300)
plt.close()
print("Figure saved to 'fet_output_characteristics.png'")


# %%
