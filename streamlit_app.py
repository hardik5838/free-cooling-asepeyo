import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Power Opti-BlackBox", layout="wide")
st.title("🧮 Power Contract Optimization (Black Box)")

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
# 2. STREAMLIT INTERFACE (INPUTS)
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Data Feed")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    st.header("2. Rates Configuration")
    
    # A. Fixed Base Rate
    base_rate_input = st.number_input("Base Rate (Fixed) €/Year", value=0.0, step=10.0, 
                                      help="Any fixed annual management fee per contract.")
    
    # B. Period Rates
    st.subheader("Period Rates (Fixed Term)")
    st.caption("Price per kW per Year (€/kW/yr)")
    default_rates = pd.DataFrame({
        'Period': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'Rate (€/kW)': [30.0, 25.0, 15.0, 12.0, 8.0, 4.0]
    })
    edited_rates = st.data_editor(default_rates, hide_index=True)
    rate_map = dict(zip(edited_rates['Period'], edited_rates['Rate (€/kW)']))
    
    # C. Penalty Factor (THE OVERRIDE)
    st.subheader("Penalty Factor")
    penalty_input = st.number_input(
        "Excess Penalty Cost (€ per Excess kW)", 
        value=1.50, # Set a reasonable market default
        step=0.10,
        format="%.2f",
        help="Cost for every kW that exceeds 105% of contracted power."
    )
    
    run_btn = st.button("RUN OPTIMIZATION", type="primary")

# -----------------------------------------------------------------------------
# 3. EXECUTION LOGIC
# -----------------------------------------------------------------------------
if uploaded_file and run_btn:
    df, col_map = load_data(uploaded_file)
    
    results = []
    
    # Progress Bar
    bar = st.progress(0)
    cups_list = df['CUPS'].unique()
    
    for idx, cups in enumerate(cups_list):
        df_c = df[df['CUPS'] == cups]
        
        # 1. Prepare Data
        max_dict = {}
        curr_powers = {}
        for i in range(1, 7):
            max_dict[i] = df_c[col_map[f'max_p{i}']]
            # Get current contract (mode or max to avoid 0s)
            curr_powers[i] = df_c[col_map[f'con_p{i}']].max()
            if curr_powers[i] == 0: curr_powers[i] = 1.0 
            
        # 2. Calculate Current Cost
        current_cost = calculate_scenario_cost(curr_powers, max_dict, rate_map, base_rate_input, penalty_input)
        
        # 3. Optimize (Two Loop Strategy)
        best_powers = {}
        prev_p = 0
        
        for i in range(1, 7):
            peak = max_dict[i].max()
            if peak == 0: peak = prev_p 
            
            # Constraint: Cannot be lower than previous period
            min_limit = max(prev_p, 0)
            
            # --- LOOP 1: COARSE (10% steps) ---
            # Search from Min Limit to 200% of Peak
            start_c = min_limit
            end_c = max(min_limit, peak * 2.0)
            step_c = max(1.0, peak * 0.1)
            candidates_1 = np.arange(start_c, end_c + step_c, step_c)
            
            best_c = start_c
            min_cost_c = float('inf')
            
            def period_cost_fn(p, period_idx):
                f_cost = p * rate_map[f'P{period_idx}']
                limit = 1.05 * p
                excess = max_dict[period_idx][max_dict[period_idx] > limit] - limit
                p_cost = excess.sum() * penalty_input
                return f_cost + p_cost

            for p in candidates_1:
                c = period_cost_fn(p, i)
                if c < min_cost_c:
                    min_cost_c = c
                    best_c = p
            
            # --- LOOP 2: FINE (1% steps around best coarse) ---
            range_fine = peak * 0.15
            start_f = max(min_limit, best_c - range_fine)
            end_f = best_c + range_fine
            step_f = max(0.1, peak * 0.01)
            
            candidates_2 = np.arange(start_f, end_f, step_f)
            
            best_f = best_c
            min_cost_f = min_cost_c
            
            for p in candidates_2:
                c = period_cost_fn(p, i)
                if c < min_cost_f:
                    min_cost_f = c
                    best_f = p
            
            best_powers[i] = round(best_f, 2)
            prev_p = best_powers[i] # Set constraint for next period
            
        # 4. Final Calc
        opt_cost = calculate_scenario_cost(best_powers, max_dict, rate_map, base_rate_input, penalty_input)
        
        # 5. Safety Option (+5%)
        safe_powers = {k: round(v * 1.05, 2) for k,v in best_powers.items()}
        safe_cost = calculate_scenario_cost(safe_powers, max_dict, rate_map, base_rate_input, penalty_input)
        
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
    res_df = pd.DataFrame(results)
    
    # Fill NaNs to prevent crashes
    res_df = res_df.fillna(0)
    res_df = res_df.sort_values('Savings', ascending=False)
    
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Total Annual Savings", f"€ {res_df['Savings'].sum():,.2f}")
    col2.metric("Centers Analyzed", len(res_df))
    
    st.subheader("🏆 Savings Leaderboard")
    
    # Prepare display dataframe (subset)
    display_df = res_df[['CUPS', 'Current_Cost', 'Optimized_Cost', 'Savings', 'Safety_Option_Cost']].copy()
    
    # Streamlit Dataframe with Formatting
    st.dataframe(
        display_df.style.format({
            'Current_Cost': '€ {:,.2f}', 
            'Optimized_Cost': '€ {:,.2f}', 
            'Savings': '€ {:,.2f}', 
            'Safety_Option_Cost': '€ {:,.2f}'
        }),
        use_container_width=True
    )
    
    st.subheader("🔍 Individual Center Analysis")
    sel_cups = st.selectbox("Select Center", res_df['CUPS'])
    
    if sel_cups:
        row = res_df[res_df['CUPS'] == sel_cups].iloc[0]
        
        # Compare Powers
        p_data = []
        for i in range(1, 7):
            p_data.append({'Period': f'P{i}', 'Type': 'Best', 'kW': row['Best_Powers'][i]})
            p_data.append({'Period': f'P{i}', 'Type': 'Safe (+5%)', 'kW': row['Safe_Powers'][i]})
            
            # Add Max Recorded for Context
            cup_raw = df[df['CUPS'] == sel_cups]
            max_val = cup_raw[col_map[f'max_p{i}']].max()
            p_data.append({'Period': f'P{i}', 'Type': 'Max Usage', 'kW': max_val})

        chart_df = pd.DataFrame(p_data)
        
        fig = px.bar(chart_df, x='Period', y='kW', color='Type', barmode='group',
                     color_discrete_map={'Best': '#00CC96', 'Safe (+5%)': '#636EFA', 'Max Usage': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)

elif not uploaded_file:
    st.info("👋 Upload a CSV to start the Black Box optimization.")
