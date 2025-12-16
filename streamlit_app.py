import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Power Opti-BlackBox (Stable)", layout="wide")
st.title("🧮 Power Contract Optimization (Black Box)")

def clean_number(x):
    """
    Robust cleaner. Handles 1.234,56 OR 1234.56
    """
    if pd.isna(x) or str(x).strip() == "": return 0.0
    if isinstance(x, (int, float)): return float(x)
    
    s = str(x).strip()
    # If "1.234,56" (Spanis) -> remove dot, replace comma
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    # If "1,23" (Spanish simple) -> replace comma
    elif ',' in s:
        s = s.replace(',', '.')
    # If "1.23" (English/Manual) -> keep as is
    
    try: return float(s)
    except: return 0.0

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Map Columns
    col_map = {}
    cols = df.columns
    for i in range(1, 7):
        # Flexible search
        max_c = [c for c in cols if f'P{i}' in c and ('Max' in c or 'max' in c)]
        con_c = [c for c in cols if f'P{i}' in c and ('Potencia' in c or 'Con' in c)]
        
        if max_c: col_map[f'max_p{i}'] = max_c[0]
        if con_c: col_map[f'con_p{i}'] = con_c[0]
    
    # Look for "Importe excesos" to get the REAL fines paid
    imp_col = [c for c in cols if 'Importe' in c and 'excesos' in c]
    if imp_col:
        col_map['importe'] = imp_col[0]
    else:
        col_map['importe'] = None
        
    targets = list(col_map.values())
    targets = [t for t in targets if t is not None]
    
    for c in targets:
        df[c] = df[c].apply(clean_number)
            
    return df, col_map

def calculate_fixed_cost(contracted_map, prices_map, base_rate):
    """Calculates ONLY the Fixed Power Term Cost."""
    cost = base_rate
    for i in range(1, 7):
        p_val = contracted_map.get(i, 0)
        cost += p_val * prices_map.get(f'P{i}', 0)
    return cost

def simulate_penalty_cost(contracted_map, max_series_map, penalty_price):
    """Simulates penalties for the Optimized scenario."""
    total_penalty = 0
    for i in range(1, 7):
        limit = 1.05 * contracted_map[i]
        series = max_series_map[i]
        # Excess calculation
        excess = series[series > limit] - limit
        total_penalty += excess.sum() * penalty_price
    return total_penalty

# -----------------------------------------------------------------------------
# 2. INPUTS (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV (Manual or Raw)", type=['csv'])
    
    st.header("2. Costs & Rates")
    
    # Input 1: Base Rate
    base_rate_input = st.number_input("Base Rate (Fixed Fee) €/Year", value=0.0, step=10.0)
    
    # Input 2: Period Rates
    st.subheader("Power Term Rates (€/kW/Year)")
    default_rates = pd.DataFrame({
        'Period': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'Rate': [30.0, 20.0, 20.0, 15.0, 15.0, 15.0]
    })
    edited_rates = st.data_editor(default_rates, hide_index=True)
    rate_map = dict(zip(edited_rates['Period'], edited_rates['Rate']))
    
    # Input 3: Penalty Factor
    st.subheader("Penalty Factor")
    penalty_input = st.number_input(
        "Estimated Penalty Cost (€/Excess kW)", 
        value=1.50, 
        step=0.1,
        help="Used to simulate future penalties."
    )
    
    run_btn = st.button("RUN OPTIMIZER", type="primary")

# -----------------------------------------------------------------------------
# 3. MAIN LOGIC
# -----------------------------------------------------------------------------
if uploaded_file and run_btn:
    df, col_map = load_data(uploaded_file)
    
    results = []
    bar = st.progress(0)
    cups_list = df['CUPS'].unique()
    
    for idx, cups in enumerate(cups_list):
        df_c = df[df['CUPS'] == cups]
        
        # A. GET CURRENT STATUS
        curr_powers = {}
        max_series = {}
        
        for i in range(1, 7):
            # Max History
            max_series[i] = df_c[col_map[f'max_p{i}']]
            # Current Contract (Take max to avoid 0s)
            val = df_c[col_map[f'con_p{i}']].max()
            curr_powers[i] = val if val > 0 else 0
        
        # 1. Calculate Fixed Cost (Current)
        curr_fixed = calculate_fixed_cost(curr_powers, rate_map, base_rate_input)
        
        # 2. Get REAL Penalty (Current)
        if col_map['importe']:
            curr_penalty = df_c[col_map['importe']].sum()
        else:
            curr_penalty = simulate_penalty_cost(curr_powers, max_series, penalty_input)
            
        total_curr = curr_fixed + curr_penalty
        
        # B. OPTIMIZATION LOOP
        best_powers = {}
        prev_p = 0
        
        for i in range(1, 7):
            peak = max_series[i].max()
            if peak == 0: peak = prev_p
            
            min_limit = max(prev_p, 0)
            
            # --- Loop 1: Coarse (10% steps) ---
            start = min_limit
            end = max(min_limit, peak * 2.5)
            step = max(1.0, peak * 0.1)
            
            candidates_1 = np.arange(start, end + step, step)
            
            best_c = start
            min_cost_c = float('inf')
            
            def check_cost(p, idx):
                f = p * rate_map[f'P{idx}']
                limit = 1.05 * p
                exc = max_series[idx][max_series[idx] > limit] - limit
                pen = exc.sum() * penalty_input
                return f + pen
            
            for p in candidates_1:
                cost = check_cost(p, i)
                if cost < min_cost_c:
                    min_cost_c = cost
                    best_c = p
                    
            # --- Loop 2: Fine (1% steps) ---
            range_fine = peak * 0.2
            start_f = max(min_limit, best_c - range_fine)
            end_f = best_c + range_fine
            step_f = max(0.1, peak * 0.01)
            
            candidates_2 = np.arange(start_f, end_f, step_f)
            best_f = best_c
            min_cost_f = min_cost_c
            
            for p in candidates_2:
                cost = check_cost(p, i)
                if cost < min_cost_f:
                    min_cost_f = cost
                    best_f = p
            
            best_powers[i] = round(best_f, 2)
            prev_p = best_powers[i]
            
        # C. FINAL RESULTS
        opt_fixed = calculate_fixed_cost(best_powers, rate_map, base_rate_input)
        opt_penalty = simulate_penalty_cost(best_powers, max_series, penalty_input)
        total_opt = opt_fixed + opt_penalty
        
        # Safety Option (+5%)
        safe_powers = {k: round(v*1.05, 2) for k,v in best_powers.items()}
        safe_fixed = calculate_fixed_cost(safe_powers, rate_map, base_rate_input)
        safe_penalty = simulate_penalty_cost(safe_powers, max_series, penalty_input)
        total_safe = safe_fixed + safe_penalty
        
        results.append({
            'CUPS': cups,
            'Current_Cost': total_curr,
            'Optimized_Cost': total_opt,
            'Safety_Cost': total_safe,
            'Savings': total_curr - total_opt,
            'Real_Penalty_Paid': curr_penalty,
            'Simulated_New_Penalty': opt_penalty,
            'Current_Powers': curr_powers,
            'Optimized_Powers': best_powers
        })
        bar.progress((idx+1)/len(cups_list))
        
    df_res = pd.DataFrame(results).sort_values('Savings', ascending=False)
    # SAFETY: Fill NaNs to prevent formatting crashes
    df_res = df_res.fillna(0.0)
    
    # -------------------------------------------------------------------------
    # 4. OUTPUTS
    # -------------------------------------------------------------------------
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Savings", f"€ {df_res['Savings'].sum():,.2f}")
    col2.metric("Avg Savings / Center", f"€ {df_res['Savings'].mean():,.2f}")
    col3.metric("Centers Analyzed", len(df_res))
    
    st.subheader("🏆 Results (Real Current Cost vs Optimized)")
    
    # SAFE DATAFRAME DISPLAY
    # We apply formatting specifically to numeric columns only
    format_dict = {
        'Current_Cost': "€ {:,.2f}",
        'Optimized_Cost': "€ {:,.2f}",
        'Savings': "€ {:,.2f}",
        'Real_Penalty_Paid': "€ {:,.2f}",
        'Simulated_New_Penalty': "€ {:,.2f}"
    }
    
    st.dataframe(
        df_res[['CUPS', 'Current_Cost', 'Optimized_Cost', 'Savings', 'Real_Penalty_Paid', 'Simulated_New_Penalty']]
        .style.format(format_dict),
        use_container_width=True
    )
    
    st.divider()
    st.subheader("🔍 Analysis Tool")
    sel_cups = st.selectbox("Select Center", df_res['CUPS'])
    
    if sel_cups:
        row = df_res[df_res['CUPS'] == sel_cups].iloc[0]
        
        # Chart
        plot_data = []
        for i in range(1, 7):
            # Current
            plot_data.append({'Period': f'P{i}', 'Type': 'Current Contract', 'kW': row['Current_Powers'][i]})
            # Optimized
            plot_data.append({'Period': f'P{i}', 'Type': 'Optimized Contract', 'kW': row['Optimized_Powers'][i]})
            # Max Usage
            c_raw = df[df['CUPS'] == sel_cups]
            mx = c_raw[col_map[f'max_p{i}']].max()
            plot_data.append({'Period': f'P{i}', 'Type': 'Max Recorded', 'kW': mx})
            
        fig = px.bar(plot_data, x='Period', y='kW', color='Type', barmode='group',
                     color_discrete_map={'Current Contract': '#EF553B', 'Optimized Contract': '#00CC96', 'Max Recorded': '#636EFA'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Cost Details for {sel_cups}:**
        - You paid **€{row['Real_Penalty_Paid']:,.2f}** in penalties (from CSV).
        - With optimization, estimated penalties drop to **€{row['Simulated_New_Penalty']:,.2f}**.
        """)

elif not uploaded_file:
    st.info("Waiting for file...")
