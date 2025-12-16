import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. THE BLACK BOX ENGINE
# -----------------------------------------------------------------------------

def clean_spanish_number(x):
    """Converts 1.234,56 -> 1234.56 robustly."""
    if pd.isna(x) or str(x).strip() == "": return 0.0
    s = str(x).replace('.', '').replace(',', '.')
    try: return float(s)
    except: return 0.0

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Auto-detect columns
    col_map = {}
    for i in range(1, 7):
        # Look for variations of Maximeter and Contracted
        cols = df.columns
        max_c = [c for c in cols if f'P{i}' in c and ('Max' in c or 'max' in c)]
        con_c = [c for c in cols if f'P{i}' in c and ('Potencia' in c or 'Con' in c)]
        if max_c: col_map[f'max_p{i}'] = max_c[0]
        if con_c: col_map[f'con_p{i}'] = con_c[0]
        
    # Clean Numbers
    targets = list(col_map.values()) + ['Importe excesos de potencia']
    for c in targets:
        if c in df.columns:
            df[c] = df[c].apply(clean_spanish_number)
            
    return df, col_map

def learn_penalty_coefficient(df, col_map):
    """
    Reverse-engineers the 'Fine' from the CSV history.
    Formula: Total_Actual_Fines / Total_Theoretical_Excess_kW
    """
    # 1. Filter rows where we actually paid a fine
    if 'Importe excesos de potencia' not in df.columns:
        return 1.5 # Fallback default multiplier if no data
        
    penalties = df[df['Importe excesos de potencia'] > 0].copy()
    
    if penalties.empty:
        return 2.0 # Default High Penalty Factor if no history found
        
    # 2. Calculate the 'Excess kW' that likely caused these fines
    total_excess_kw_accum = 0
    
    for i in range(1, 7):
        max_col = col_map.get(f'max_p{i}')
        con_col = col_map.get(f'con_p{i}')
        if max_col and con_col:
            # Excess Rule: Max - 1.05 * Contracted
            # We assume the CSV Maximeter is the billing determinant
            excess = penalties[max_col] - (1.05 * penalties[con_col])
            # Only count positive excess
            total_excess_kw_accum += excess.apply(lambda x: x if x > 0 else 0).sum()
            
    total_fines_paid = penalties['Importe excesos de potencia'].sum()
    
    if total_excess_kw_accum > 0:
        # This is the "Implied Price per kW of Excess"
        return total_fines_paid / total_excess_kw_accum
    else:
        return 10.0 # High default if data is weird

def calculate_scenario_cost(contracted_p1_p6, max_series_dict, prices_p1_p6, base_rate, penalty_coef):
    """
    Calculates Total Cost = (Fixed Power Cost) + (Simulated Fines) + Base Rate
    """
    # 1. Fixed Power Cost
    fixed_cost = 0
    for i in range(1, 7):
        fixed_cost += contracted_p1_p6[i] * prices_p1_p6[f'P{i}']
    
    fixed_cost += base_rate
    
    # 2. Simulated Fines
    # We apply the "Learned Coefficient" to any NEW simulated excess
    simulated_fine = 0
    for i in range(1, 7):
        max_vals = max_series_dict[i]
        limit = 1.05 * contracted_p1_p6[i]
        
        # Calculate excess vector
        excesses = max_vals[max_vals > limit]
        excess_kw_sum = (excesses - limit).sum()
        
        simulated_fine += excess_kw_sum * penalty_coef
        
    return fixed_cost + simulated_fine

# -----------------------------------------------------------------------------
# 2. STREAMLIT INTERFACE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Power Opti-BlackBox", layout="wide")
st.title("🧮 Power Contract Optimization (Black Box)")

# --- INPUT SECTION ---
with st.sidebar:
    st.header("1. Data Feed")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    st.header("2. Rates Input")
    st.caption("Enter the costs as requested.")
    
    # INPUT 1: BASE RATE (FIXED)
    base_rate_input = st.number_input("Base Rate (Fixed) €/Year", value=0.0, step=10.0)
    
    # INPUT 2: EACH PERIOD RATE (VARIABLE)
    # We assume this is the Power Term Price (€/kW/Year) for each period
    default_rates = pd.DataFrame({
        'Period': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'Rate (€/kW)': [30.0, 25.0, 15.0, 12.0, 8.0, 4.0]
    })
    edited_rates = st.data_editor(default_rates, hide_index=True)
    
    # Convert rates to simple dict
    rate_map = dict(zip(edited_rates['Period'], edited_rates['Rate (€/kW)']))
    
    run_btn = st.button("RUN CALCULATOR", type="primary")

# --- EXECUTION ---
if uploaded_file and run_btn:
    df, col_map = load_data(uploaded_file)
    
    # A. AUTO-LEARN PENALTIES
    penalty_factor = learn_penalty_coefficient(df, col_map)
    st.success(f"Black Box Calibration: Detected Implied Penalty Rate of **{penalty_factor:.2f} €/kW** from historical fines.")
    
    results = []
    
    # Progress Bar
    bar = st.progress(0)
    cups_list = df['CUPS'].unique()
    
    for idx, cups in enumerate(cups_list):
        # Data for this Center
        df_c = df[df['CUPS'] == cups]
        
        # 1. Prepare Data Series
        max_dict = {}
        curr_powers = {}
        for i in range(1, 7):
            max_dict[i] = df_c[col_map[f'max_p{i}']]
            # Get current contract (mode or max to avoid 0s)
            curr_powers[i] = df_c[col_map[f'con_p{i}']].max()
            if curr_powers[i] == 0: curr_powers[i] = 1.0 # Safety
            
        # 2. Calculate Current Cost (Baseline)
        current_cost = calculate_scenario_cost(curr_powers, max_dict, rate_map, base_rate_input, penalty_factor)
        
        # 3. OPTIMIZATION (The 2-Loop Strategy)
        # We optimize P1, then P2>=P1, then P3>=P2...
        
        best_powers = {}
        prev_p = 0
        
        for i in range(1, 7):
            peak = max_dict[i].max()
            if peak == 0: peak = prev_p # Fallback
            
            # --- LOOP 1: COARSE (10% Blocks) ---
            # Search space: Max(Prev_P, 0) to 200% Peak
            candidates_1 = np.arange(max(prev_p, 0), max(prev_p, peak * 2.0) + 1, max(1, peak*0.1))
            
            best_c = prev_p
            min_cost_c = float('inf')
            
            # We simplify cost function for just this period logic
            # (We only care about this period's contribution to minimize)
            def period_cost_fn(p, period_idx):
                f_cost = p * rate_map[f'P{period_idx}']
                limit = 1.05 * p
                excess = max_dict[period_idx][max_dict[period_idx] > limit] - limit
                p_cost = excess.sum() * penalty_factor
                return f_cost + p_cost

            for p in candidates_1:
                c = period_cost_fn(p, i)
                if c < min_cost_c:
                    min_cost_c = c
                    best_c = p
            
            # --- LOOP 2: FINE (1% Search around best_c) ---
            range_fine = peak * 0.15
            candidates_2 = np.arange(max(prev_p, best_c - range_fine), best_c + range_fine, max(0.1, peak*0.01))
            
            best_f = best_c
            min_cost_f = min_cost_c
            
            for p in candidates_2:
                c = period_cost_fn(p, i)
                if c < min_cost_f:
                    min_cost_f = c
                    best_f = p
            
            # Store and set constraint for next
            best_powers[i] = round(best_f, 2)
            prev_p = best_powers[i]
            
        # 4. Final Cost Calculation
        opt_cost = calculate_scenario_cost(best_powers, max_dict, rate_map, base_rate_input, penalty_factor)
        
        # 5. Anomaly Detection (2nd Best Option)
        # We generate a "Safety" option (Best + 5%)
        safe_powers = {k: v * 1.05 for k,v in best_powers.items()}
        safe_cost = calculate_scenario_cost(safe_powers, max_dict, rate_map, base_rate_input, penalty_factor)
        
        results.append({
            'CUPS': cups,
            'Current_Cost': current_cost,
            'Optimized_Cost': opt_cost,
            'Safety_Option_Cost': safe_cost,
            'Savings': current_cost - opt_cost,
            'Best_Powers': best_powers,
            'Safe_Powers': safe_powers
        })
        
        bar.progress((idx+1)/len(cups_list))
        
    # --- OUTPUTS ---
    res_df = pd.DataFrame(results).sort_values('Savings', ascending=False)
    
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Total Annual Savings", f"€ {res_df['Savings'].sum():,.2f}")
    col2.metric("Centers Analyzed", len(res_df))
    
    st.subheader("1. Savings Leaderboard")
    st.dataframe(
        res_df[['CUPS', 'Current_Cost', 'Optimized_Cost', 'Savings', 'Safety_Option_Cost']]
        .style.format("€ {:,.2f}"), 
        use_container_width=True
    )
    
    st.subheader("2. Detailed Center Analysis")
    sel_cups = st.selectbox("Select Center", res_df['CUPS'])
    
    if sel_cups:
        row = res_df[res_df['CUPS'] == sel_cups].iloc[0]
        
        # Display Powers Table
        st.write("#### Optimized Power Configuration")
        p_data = []
        for i in range(1, 7):
            p_data.append({
                'Period': f'P{i}',
                'Best Option (kW)': row['Best_Powers'][i],
                'Safety Option (kW)': row['Safe_Powers'][i]
            })
        st.table(pd.DataFrame(p_data))
        
        st.write(f"**Calculated Fine Rate:** €{penalty_factor:.2f} per excess kW (Derived from history)")

elif not uploaded_file:
    st.info("Waiting for CSV upload...")
