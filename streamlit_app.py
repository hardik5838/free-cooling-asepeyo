import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Power Optimization Engine", layout="wide")

def clean_spanish_number(x):
    """Robustly converts Spanish format (1.234,56) to float."""
    if pd.isna(x) or x == "":
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    
    clean_str = str(x).strip()
    # Remove thousand separators (.) and replace decimal (,) with (.)
    clean_str = clean_str.replace('.', '').replace(',', '.')
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file)
    
    # Dynamic column identification
    # We look for columns like "Maxímetro P1" or similar
    cols_map = {}
    for i in range(1, 7):
        # Find Maximeter col
        max_col = [c for c in df.columns if f'P{i}' in c and ('Max' in c or 'max' in c)]
        # Find Contracted col
        con_col = [c for c in df.columns if f'P{i}' in c and ('Potencia' in c or 'Con' in c)]
        
        if max_col: cols_map[f'max_p{i}'] = max_col[0]
        if con_col: cols_map[f'con_p{i}'] = con_col[0]

    # Clean numeric columns
    for col in cols_map.values():
        df[col] = df[col].apply(clean_spanish_number)
        
    return df, cols_map

# -----------------------------------------------------------------------------
# 2. CORE LOGIC
# -----------------------------------------------------------------------------

def calculate_cost_components(contracted_kw, max_readings_series, fixed_price_kw_yr, penalty_price_kwh):
    """
    Returns: (Fixed_Cost_Total, Penalty_Cost_Total)
    Assumption: The dataset passed (max_readings_series) represents 1 Year of data.
    """
    # 1. Fixed Cost (Termino Potencia)
    # Price is €/kW/Year. We apply this ONCE for the annual analysis.
    fixed_cost = contracted_kw * fixed_price_kw_yr
    
    # 2. Penalty Cost (Excesos)
    # Rule: If Max > 1.05 * Contracted -> Pay penalty on difference
    threshold = 1.05 * contracted_kw
    
    # Filter only periods where demand exceeded threshold
    excesses = max_readings_series[max_readings_series > threshold]
    
    # Total Excess kW (sum of all months)
    total_excess_kw = (excesses - threshold).sum()
    
    penalty_cost = total_excess_kw * penalty_price_kwh
    
    return fixed_cost, penalty_cost

def optimize_p_n(max_series, fixed_price, penalty_price, min_limit_kw):
    """
    Finds optimal Power for ONE period (Pn), respecting Pn >= min_limit_kw
    using the 2-Loop (Coarse + Fine) strategy.
    """
    peak = max_series.max() if not max_series.empty else 0
    if peak == 0: return max(min_limit_kw, 0) # No usage
    
    # Define Cost Function for Minimization
    def get_total_cost(p_cand):
        f, p = calculate_cost_components(p_cand, max_series, fixed_price, penalty_price)
        return f + p

    # --- LOOP 1: COARSE SEARCH (0% to 200% of Peak, 5% steps) ---
    step_coarse = max(1.0, peak * 0.05) 
    start = max(min_limit_kw, 0) # Must be at least previous period power
    end = max(min_limit_kw, peak * 2.0)
    
    search_space_1 = np.arange(start, end + step_coarse, step_coarse)
    
    best_p_coarse = start
    min_cost_coarse = float('inf')
    
    for p in search_space_1:
        cost = get_total_cost(p)
        if cost < min_cost_coarse:
            min_cost_coarse = cost
            best_p_coarse = p
            
    # --- LOOP 2: FINE SEARCH (+/- 10% around best coarse, 0.5% steps) ---
    step_fine = max(0.1, peak * 0.005)
    start_fine = max(min_limit_kw, best_p_coarse - (peak * 0.1))
    end_fine = best_p_coarse + (peak * 0.1)
    
    search_space_2 = np.arange(start_fine, end_fine, step_fine)
    
    best_p_final = best_p_coarse
    min_cost_final = min_cost_coarse
    
    for p in search_space_2:
        cost = get_total_cost(p)
        if cost < min_cost_final:
            min_cost_final = cost
            best_p_final = p
            
    return round(best_p_final, 2)

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI
# -----------------------------------------------------------------------------

st.sidebar.title("⚙️ Optimization Settings")

# A. File Upload
st.sidebar.subheader("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_raw, cols_map = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"Loaded {len(df_raw)} rows.")
else:
    st.info("Please upload your CSV file.")
    st.stop()

# B. Cost Configuration (Editable Grid)
st.sidebar.subheader("2. Cost Parameters")
st.sidebar.info("Enter the **Fixed Power Price** (Termino Potencia) and **Penalty Price**.")

# Default values updated to be more realistic for Spain 2024/25
# Power Term is usually €/kW/Year (e.g. ~30-60€ for P1, ~5-10€ for P6)
default_costs = {
    'Period': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
    'Fixed_Price_€_kW_Year': [50.0, 40.0, 20.0, 15.0, 8.0, 4.0],  # User "Base Contract Rate"
    'Penalty_Price_€_kWh': [0.60, 0.50, 0.40, 0.30, 0.20, 0.10]   # User "Peak Penalties"
}
df_costs_input = pd.DataFrame(default_costs)
edited_costs = st.sidebar.data_editor(df_costs_input, hide_index=True)

# Convert costs to dictionary for fast lookup
costs_dict = edited_costs.set_index('Period').to_dict('index')

# C. Run Button
if st.button("Calculate Optimization"):
    
    results = []
    
    # Process each CUPS
    unique_cups = df_raw['CUPS'].unique()
    
    progress_bar = st.progress(0)
    
    for idx, cups in enumerate(unique_cups):
        df_cups = df_raw[df_raw['CUPS'] == cups]
        
        # 1. Get Current Status (and fix the 0kW bug)
        curr_powers = {}
        curr_fixed_cost = 0
        curr_penalty_cost = 0
        
        # Extract max usage history for this CUPS
        max_series_dict = {}
        for i in range(1, 7):
             max_series_dict[i] = df_cups[cols_map.get(f'max_p{i}')]

        # Calculate CURRENT COSTS based on what is in the file
        for i in range(1, 7):
            # Safe extraction of current contracted power (handle 0s)
            p_col = cols_map.get(f'con_p{i}')
            if p_col:
                # We take the MAX contracted value found to avoid the "0" bug 
                # (assuming contract doesn't change drastically mid-year)
                val = df_cups[p_col].max()
                curr_p = val if val > 0 else 0
            else:
                curr_p = 0
                
            curr_powers[i] = curr_p
            
            # Cost Calc
            f_cost, p_cost = calculate_cost_components(
                curr_p, 
                max_series_dict[i], 
                costs_dict[f'P{i}']['Fixed_Price_€_kW_Year'],
                costs_dict[f'P{i}']['Penalty_Price_€_kWh']
            )
            curr_fixed_cost += f_cost
            curr_penalty_cost += p_cost

        # 2. Optimize (Sequential P1 -> P6)
        opt_powers = {}
        opt_fixed_cost = 0
        opt_penalty_cost = 0
        last_p = 0 # Constraint P(n) >= P(n-1)
        
        for i in range(1, 7):
            # Optimize this period
            best_p = optimize_p_n(
                max_series_dict[i],
                costs_dict[f'P{i}']['Fixed_Price_€_kW_Year'],
                costs_dict[f'P{i}']['Penalty_Price_€_kWh'],
                last_p # Min limit
            )
            
            opt_powers[i] = best_p
            last_p = best_p # Update floor for next period
            
            # Calculate Optimized Costs
            f_cost, p_cost = calculate_cost_components(
                best_p,
                max_series_dict[i],
                costs_dict[f'P{i}']['Fixed_Price_€_kW_Year'],
                costs_dict[f'P{i}']['Penalty_Price_€_kWh']
            )
            opt_fixed_cost += f_cost
            opt_penalty_cost += p_cost
            
        # 3. Compile Results
        total_curr = curr_fixed_cost + curr_penalty_cost
        total_opt = opt_fixed_cost + opt_penalty_cost
        
        res_row = {
            'CUPS': cups,
            'Current_Total_€': total_curr,
            'Optimized_Total_€': total_opt,
            'Savings_€': total_curr - total_opt,
            'Current_Penalties_€': curr_penalty_cost, # Debug column
            'Optimized_Penalties_€': opt_penalty_cost # Debug column
        }
        # Add power details
        for i in range(1, 7):
            res_row[f'P{i}_New'] = opt_powers[i]
            res_row[f'P{i}_Old'] = curr_powers[i]
            
        results.append(res_row)
        progress_bar.progress((idx + 1) / len(unique_cups))
        
    df_res = pd.DataFrame(results)
    
    # -------------------------------------------------------------------------
    # 4. OUTPUTS & VISUALIZATION
    # -------------------------------------------------------------------------
    st.divider()
    
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    total_sav = df_res['Savings_€'].sum()
    col1.metric("Total Annual Savings", f"€ {total_sav:,.2f}")
    col2.metric("Centers Optimized", len(unique_cups))
    
    # Detailed Table
    st.subheader("📋 Results Leaderboard (Best Savings)")
    st.dataframe(
        df_res[['CUPS', 'Current_Total_€', 'Optimized_Total_€', 'Savings_€', 'Current_Penalties_€', 'P1_New', 'P6_New']]
        .sort_values('Savings_€', ascending=False)
        .style.format("€ {:,.2f}", subset=['Current_Total_€', 'Optimized_Total_€', 'Savings_€', 'Current_Penalties_€']),
        use_container_width=True
    )
    
    # Dril Down
    st.divider()
    st.subheader("🔍 Deep Dive: Why are we saving?")
    selected_cup = st.selectbox("Select Center to Analyze", df_res['CUPS'].unique())
    
    if selected_cup:
        row = df_res[df_res['CUPS'] == selected_cup].iloc[0]
        
        # A. Cost Breakdown Chart (Stacked)
        st.write("### 1. Cost Structure Comparison")
        
        # Reconstruct detailed breakdown for chart
        breakdown_data = [
            {'Type': 'Current', 'Category': 'Fixed Cost', 'Amount': row['Current_Total_€'] - row['Current_Penalties_€']},
            {'Type': 'Current', 'Category': 'Penalties', 'Amount': row['Current_Penalties_€']},
            {'Type': 'Optimized', 'Category': 'Fixed Cost', 'Amount': row['Optimized_Total_€'] - row['Optimized_Penalties_€']},
            {'Type': 'Optimized', 'Category': 'Penalties', 'Amount': row['Optimized_Penalties_€']},
        ]
        fig_cost = px.bar(
            breakdown_data, x='Type', y='Amount', color='Category', 
            title="Where is the money going? (Fixed vs Penalties)",
            color_discrete_map={'Fixed Cost': '#636EFA', 'Penalties': '#EF553B'}
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # B. Power Comparison Chart
        st.write("### 2. Contracted Power Changes")
        
        # Prep data
        power_data = []
        for i in range(1, 7):
            power_data.append({'Period': f'P{i}', 'Version': 'Old Contract', 'kW': row[f'P{i}_Old']})
            power_data.append({'Period': f'P{i}', 'Version': 'New Optimized', 'kW': row[f'P{i}_New']})
            
            # Add Max usage for context
            # (Need to fetch from raw df again)
            cup_raw = df_raw[df_raw['CUPS'] == selected_cup]
            max_val = cup_raw[cols_map.get(f'max_p{i}')].max()
            power_data.append({'Period': f'P{i}', 'Version': 'Max Recorded Usage', 'kW': max_val})

        fig_pow = px.bar(
            power_data, x='Period', y='kW', color='Version', barmode='group',
            color_discrete_map={'Old Contract': '#EF553B', 'New Optimized': '#00CC96', 'Max Recorded Usage': 'Gray'}
        )
        st.plotly_chart(fig_pow, use_container_width=True)
        
        st.info("""
        **Note on Inputs:** - **Fixed Price:** Ensure you entered the ANNUAL price per kW (e.g., €40/kW/Year).
        - **Penalty Price:** This estimates the surcharge for exceeding 105% of contracted power.
        """)
