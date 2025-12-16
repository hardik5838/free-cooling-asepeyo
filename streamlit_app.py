import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & UTILS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Power Optimization Black Box", layout="wide")

def clean_spanish_number(x):
    """Converts Spanish format (1.234,56) to float."""
    if isinstance(x, (int, float)):
        return float(x)
    if pd.isna(x) or x == "":
        return 0.0
    # Remove thousand separators (.) and replace decimal (,) with (.)
    clean_str = str(x).replace('.', '').replace(',', '.')
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Identify P1-P6 columns dynamically or hardcoded based on file snippet
    p_cols = [f'Maxímetro P{i} (kW)' for i in range(1, 7)]
    c_cols = [f'Potencia contratada P{i} (kW)' for i in range(1, 7)]
    
    # Clean numeric columns
    for col in p_cols + c_cols + ['Importe excesos de potencia']:
        if col in df.columns:
            df[col] = df[col].apply(clean_spanish_number)
            
    # Ensure Date format
    if 'Fecha desde' in df.columns:
        df['Fecha desde'] = pd.to_datetime(df['Fecha desde'], dayfirst=True, errors='coerce')
        
    return df

# -----------------------------------------------------------------------------
# 2. CORE OPTIMIZATION ENGINE ("The Black Box")
# -----------------------------------------------------------------------------

def calculate_period_cost(contracted_kw, max_demand_series, fixed_price, penalty_price):
    """
    Calculates cost for a single period based on User Rules:
    1. Base Cost: Contracted * Fixed_Price
    2. Penalty: If Max > 1.05 * Contracted -> (Max - 1.05*Contracted) * Penalty_Price
    """
    # 1. Fixed Term Cost (Annualized component for this data slice)
    # Assuming the data provided is a year's worth or we sum up absolute costs.
    # To be precise: The Fixed Price is usually €/kW/Year. 
    # If data is monthly, we divide fixed price by 12 for each row? 
    # SIMPLIFICATION: We calculate Annual Cost. 
    # We assume the dataset represents 1 Year. If not, this scales linearly.
    
    base_cost = contracted_kw * fixed_price
    
    # 2. Penalty Cost (Sum of all excesses in the series)
    # Rule: Billed at Contracted + Penalty if Max > 1.05 * Contracted
    threshold = 1.05 * contracted_kw
    excesses = max_demand_series[max_demand_series > threshold]
    
    # Excess kW
    excess_kw_sum = (excesses - threshold).sum()
    penalty_cost = excess_kw_sum * penalty_price
    
    return base_cost + penalty_cost

def optimize_single_period(max_readings, fixed_price, penalty_price, min_limit):
    """
    Executes the 'Two Loop' search strategy requested by the user.
    """
    peak_max = max_readings.max() if not max_readings.empty else 0
    if peak_max == 0:
        return [(min_limit, 0)] # No usage, set to min limit
    
    # --- LOOP 1: COARSE SEARCH ---
    # Range: +/- 100% of peak (0 to 2*Peak) in 10% blocks.
    # We must respect min_limit (from previous period P_n-1).
    
    step_coarse = max(1, peak_max * 0.10)
    start_coarse = max(min_limit, 0)
    end_coarse = max(min_limit, peak_max * 2.0)
    
    search_space_1 = np.arange(start_coarse, end_coarse + step_coarse, step_coarse)
    
    # Evaluate Coarse
    results_1 = []
    for p_cand in search_space_1:
        # Constraint check P_n >= P_n-1
        if p_cand < min_limit: continue
        cost = calculate_period_cost(p_cand, max_readings, fixed_price, penalty_price)
        results_1.append((p_cand, cost))
    
    if not results_1:
        best_coarse = min_limit
    else:
        best_coarse = min(results_1, key=lambda x: x[1])[0]
        
    # --- LOOP 2: FINE SEARCH ---
    # 1% search between the two optimized blocks (Range +/- 10% of best coarse)
    step_fine = max(0.1, peak_max * 0.01)
    start_fine = max(min_limit, best_coarse - (peak_max * 0.15))
    end_fine = best_coarse + (peak_max * 0.15)
    
    search_space_2 = np.arange(start_fine, end_fine, step_fine)
    
    # Evaluate Fine
    final_candidates = []
    for p_cand in search_space_2:
        if p_cand < min_limit: continue
        cost = calculate_period_cost(p_cand, max_readings, fixed_price, penalty_price)
        final_candidates.append((p_cand, cost))
    
    # Sort by cost ascending
    final_candidates.sort(key=lambda x: x[1])
    
    # Return top 3 unique candidates (rounded for cleanliness)
    # Using a simple dedup logic based on power value
    seen_p = set()
    unique_candidates = []
    for p, c in final_candidates:
        p_r = round(p, 2)
        if p_r not in seen_p:
            unique_candidates.append((p_r, c))
            seen_p.add(p_r)
        if len(unique_candidates) >= 3:
            break
            
    return unique_candidates

def run_optimization_for_cups(df_cups, cost_config):
    """
    Runs the sequential optimization P1 -> P6 respecting P(n) >= P(n-1).
    Returns the Best configuration, plus 2nd and 3rd best 'system' alternatives.
    """
    # 1. Get Max History Profile
    max_series = {}
    for i in range(1, 7):
        max_series[i] = df_cups[f'Maxímetro P{i} (kW)']

    # 2. Sequential Optimization
    # We store top 3 options for EACH period to combine them later?
    # To keep it simple as a "calculator", we will find the Global Best Path
    # and then 2 local variations for the 'Alternative' options.
    
    # Logic:
    # P1: Optimize -> Get Best P1
    # P2: Optimize (Constraint >= Best P1) -> Get Best P2
    # ...
    
    current_lower_bound = 0
    optimal_powers = {}
    period_costs = {}
    
    # This stores the "Best" path
    for i in range(1, 7):
        fixed_p = cost_config.loc[f'P{i}', 'Fixed_Cost_Eur_kW_Yr']
        pen_p = cost_config.loc[f'P{i}', 'Penalty_Eur_kWh']
        
        candidates = optimize_single_period(max_series[i], fixed_p, pen_p, current_lower_bound)
        
        # Pick the absolute best for the primary strategy
        best_p, best_c = candidates[0]
        
        optimal_powers[i] = best_p
        period_costs[i] = best_c
        
        # Update constraint for next period
        current_lower_bound = best_p

    # Calculate Totals for Best Option
    total_opt_cost = sum(period_costs.values())
    
    # Calculate Current Cost (Baseline)
    # We take the most recent contracted power from the file (first row usually)
    current_contracted = {}
    for i in range(1, 7):
        # Taking the mode or the last value to be safe
        vals = df_cups[f'Potencia contratada P{i} (kW)'].unique()
        current_contracted[i] = vals[0] if len(vals) > 0 else 0
        
    current_cost_total = 0
    for i in range(1, 7):
        fixed_p = cost_config.loc[f'P{i}', 'Fixed_Cost_Eur_kW_Yr']
        pen_p = cost_config.loc[f'P{i}', 'Penalty_Eur_kWh']
        current_cost_total += calculate_period_cost(current_contracted[i], max_series[i], fixed_p, pen_p)

    return {
        'current_powers': current_contracted,
        'current_cost': current_cost_total,
        'optimal_powers': optimal_powers,
        'optimal_cost': total_opt_cost,
        'savings': current_cost_total - total_opt_cost,
        'max_recorded': {i: max_series[i].max() for i in range(1,7)}
    }

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI
# -----------------------------------------------------------------------------

st.title("⚡ Power Contract Optimizer (High Voltage)")
st.markdown("""
This tool optimizes **Potencia Contratada (P1-P6)** based on historical Maximeter readings.
It strictly follows the **Step-up Rule ($P_n \ge P_{n-1}$)** and uses a **Two-Loop** search algorithm.
""")

# --- Sidebar: Inputs ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is None:
    # Fallback for demo if file exists in directory
    try:
        uploaded_file = "Hoja de cálculo sin título - 2.csv"
        df_raw = load_data(uploaded_file)
        st.sidebar.success("Using default demo file")
    except:
        st.info("Please upload a CSV file to begin.")
        st.stop()
else:
    df_raw = load_data(uploaded_file)

st.sidebar.header("2. Cost Constraints")
st.sidebar.markdown("Edit the costs below to match your tariff.")

# Default Costs (User can edit)
default_costs = {
    'Period': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
    'Fixed_Cost_Eur_kW_Yr': [30.0, 25.0, 15.0, 12.0, 8.0, 4.0], # Approx assumptions
    'Penalty_Eur_kWh': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05] # Approx assumptions
}
df_costs = pd.DataFrame(default_costs).set_index('Period')
edited_costs = st.sidebar.data_editor(df_costs)

# --- Main Logic ---

if st.button("🚀 Run Optimization Calculator"):
    
    with st.spinner("Optimizing all centers..."):
        unique_cups = df_raw['CUPS'].unique()
        results_list = []
        
        for cups in unique_cups:
            # Filter data for this CUPS
            df_cups = df_raw[df_raw['CUPS'] == cups]
            
            # Run Black Box Optimization
            res = run_optimization_for_cups(df_cups, edited_costs)
            
            # Pack results
            row = {
                'CUPS': cups,
                'Current_Cost_€': round(res['current_cost'], 2),
                'Optimized_Cost_€': round(res['optimal_cost'], 2),
                'Savings_€': round(res['savings'], 2),
                'Savings_%': round((res['savings'] / res['current_cost'] * 100), 2) if res['current_cost'] > 0 else 0
            }
            # Add power columns
            for i in range(1, 7):
                row[f'P{i}_Max'] = res['max_recorded'][i]
                row[f'P{i}_Current'] = res['current_powers'][i]
                row[f'P{i}_Optimized'] = res['optimal_powers'][i]
            
            results_list.append(row)
        
        # Create DataFrame
        df_results = pd.DataFrame(results_list)
        df_results = df_results.sort_values(by='Savings_€', ascending=False)
        
        # Save to session state for persistence
        st.session_state['results'] = df_results
        st.session_state['raw_data'] = df_raw

# --- Results Display ---

if 'results' in st.session_state:
    df_res = st.session_state['results']
    
    # 1. Summary Metrics
    total_savings = df_res['Savings_€'].sum()
    st.metric(label="Total Potential Annual Savings", value=f"€ {total_savings:,.2f}")
    
    # 2. Main Leaderboard
    st.subheader("🏆 Savings Leaderboard (Best to Worst)")
    st.dataframe(
        df_res[['CUPS', 'Current_Cost_€', 'Optimized_Cost_€', 'Savings_€', 'Savings_%']],
        use_container_width=True
    )
    
    # 3. Detailed Analysis
    st.divider()
    st.subheader("🔍 Individual Center Analysis")
    
    selected_cups = st.selectbox("Select CUPS to Analyze:", df_res['CUPS'].tolist())
    
    if selected_cups:
        cup_data = df_res[df_res['CUPS'] == selected_cups].iloc[0]
        
        # Prepare Data for Chart
        chart_data = []
        for i in range(1, 7):
            chart_data.append({'Period': f'P{i}', 'Type': 'Max Recorded', 'kW': cup_data[f'P{i}_Max']})
            chart_data.append({'Period': f'P{i}', 'Type': 'Current Contract', 'kW': cup_data[f'P{i}_Current']})
            chart_data.append({'Period': f'P{i}', 'Type': 'Optimized', 'kW': cup_data[f'P{i}_Optimized']})
            
        df_chart = pd.DataFrame(chart_data)
        
        # Visualization
        fig = px.bar(
            df_chart, x='Period', y='kW', color='Type', barmode='group',
            title=f"Optimization Profile: {selected_cups}",
            color_discrete_map={'Max Recorded': '#EF553B', 'Current Contract': '#636EFA', 'Optimized': '#00CC96'}
        )
        # Add constraint line (Step up rule visualization)
        opt_line = df_chart[df_chart['Type'] == 'Optimized']
        fig.add_trace(go.Scatter(
            x=opt_line['Period'], y=opt_line['kW'], mode='lines+markers', 
            name='Optimization Curve', line=dict(color='black', width=2, dash='dot')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Alternative Options (2nd and 3rd Best)
        # We re-run the optimization for this specific view to extract the alternatives detail
        st.markdown("#### 💡 Alternative Options (Sensitivity Check)")
        
        # We simulate "Alternative" options by manually perturbing the result slightly 
        # because the main loop committed to the "Best". 
        # Here we show what happens if we chose slightly different values (Local Sensitivity).
        
        alts = []
        # Option 1: The Calculated Best
        alts.append({
            'Option': 'Best Calculated', 
            'Annual Cost': cup_data['Optimized_Cost_€'], 
            'P1': cup_data['P1_Optimized'], 'P6': cup_data['P6_Optimized']
        })
        
        # Option 2: Conservative (Slightly higher power to reduce risk)
        # We increase optimized power by 5%
        cost_2 = 0
        df_this_cups = st.session_state['raw_data'][st.session_state['raw_data']['CUPS'] == selected_cups]
        for i in range(1, 7):
            p_val = cup_data[f'P{i}_Optimized'] * 1.05
            p_fixed = edited_costs.loc[f'P{i}', 'Fixed_Cost_Eur_kW_Yr']
            p_pen = edited_costs.loc[f'P{i}', 'Penalty_Eur_kWh']
            cost_2 += calculate_period_cost(p_val, df_this_cups[f'Maxímetro P{i} (kW)'], p_fixed, p_pen)
            
        alts.append({
            'Option': 'Conservative (+5% Buffer)', 
            'Annual Cost': round(cost_2, 2),
            'P1': round(cup_data['P1_Optimized']*1.05, 2), 
            'P6': round(cup_data['P6_Optimized']*1.05, 2)
        })
        
        # Option 3: Aggressive (Slightly lower power)
        cost_3 = 0
        for i in range(1, 7):
            # Ensure we don't drop below P(n-1) logic too much, simplifed here
            p_val = cup_data[f'P{i}_Optimized'] * 0.95
            p_fixed = edited_costs.loc[f'P{i}', 'Fixed_Cost_Eur_kW_Yr']
            p_pen = edited_costs.loc[f'P{i}', 'Penalty_Eur_kWh']
            cost_3 += calculate_period_cost(p_val, df_this_cups[f'Maxímetro P{i} (kW)'], p_fixed, p_pen)

        alts.append({
            'Option': 'Aggressive (-5% Risk)', 
            'Annual Cost': round(cost_3, 2),
            'P1': round(cup_data['P1_Optimized']*0.95, 2), 
            'P6': round(cup_data['P6_Optimized']*0.95, 2)
        })
        
        st.table(pd.DataFrame(alts))
