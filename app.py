import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Reliarisk FlowCast",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton>button {
        width: 100%;
        background-color: #004B87; 
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #004B87;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MOTOR ESTADÍSTICO CON TRUNCAMIENTO (BACKEND) ---

def get_scipy_dist(dist_type, params):
    """
    Convierte parámetros de usuario a objetos de distribución congelados (frozen) de Scipy.
    Retorna el objeto rv de scipy.
    """
    if dist_type == 'Normal':
        return stats.norm(loc=params['mean'], scale=params['std'])
    
    elif dist_type == 'Lognormal':
        # Conversión de Media/DesvEst aritmética a Mu/Sigma logarítmicos
        # mu = ln(mean) - 0.5 * ln(1 + var/mean^2)
        # sigma = sqrt(ln(1 + var/mean^2))
        mean, std = params['mean'], params['std']
        # Evitar errores con 0
        if mean <= 0: mean = 0.001
        
        var = std**2
        sigma_log = np.sqrt(np.log(1 + (var / mean**2)))
        mu_log = np.log(mean) - 0.5 * sigma_log**2
        scale = np.exp(mu_log)
        # En scipy lognorm: s=sigma, scale=exp(mu)
        return stats.lognorm(s=sigma_log, scale=scale)
    
    elif dist_type == 'Weibull':
        # Scipy weibull_min: c=shape (k), scale=scale (lambda)
        return stats.weibull_min(c=params['shape'], scale=params['scale'])
    
    elif dist_type == 'Gamma':
        # Scipy gamma: a=shape (alpha), scale=scale (theta)
        return stats.gamma(a=params['shape'], scale=params['scale'])
    
    return None

def generate_samples(dist_type, params, n_iter):
    """
    Genera muestras aleatorias considerando truncamiento por Transformada Inversa.
    """
    # 1. Distribuciones naturalmente acotadas (No requieren truncamiento externo complejo)
    if dist_type == 'BetaPERT':
        mn, md, mx = params['min'], params['mode'], params['max']
        alpha = (4 * md + mx - 5 * mn) / (mx - mn)
        beta = (5 * mx - mn - 4 * md) / (mx - mn)
        return mn + (mx - mn) * np.random.beta(alpha, beta, n_iter)
    
    elif dist_type == 'Triangular':
        return np.random.triangular(params['min'], params['mode'], params['max'], n_iter)
    
    elif dist_type == 'Uniforme':
        return np.random.uniform(params['min'], params['max'], n_iter)
    
    elif dist_type == 'Determinístico':
        return np.full(n_iter, params['value'])

    # 2. Distribuciones abiertas/semi-abiertas que requieren Truncamiento (Normal, LogN, Weibull, Gamma)
    else:
        # Obtener objeto de distribución de Scipy
        rv = get_scipy_dist(dist_type, params)
        if rv is None: return np.zeros(n_iter)

        # Verificar si el usuario activó truncamiento
        trunc_min = params.get('trunc_min', -np.inf)
        trunc_max = params.get('trunc_max', np.inf)

        # Si no hay truncamiento efectivo, usar rvs directo (más rápido)
        if trunc_min == -np.inf and trunc_max == np.inf:
            return rv.rvs(size=n_iter)
        
        # Lógica de Truncamiento (Inverse Transform Sampling)
        # a. Calcular probabilidades acumuladas (CDF) de los cortes
        try:
            p_min = rv.cdf(trunc_min)
            p_max = rv.cdf(trunc_max)
        except:
            return rv.rvs(size=n_iter) # Fallback si falla cálculo

        # Validar consistencia
        if p_max <= p_min:
            # Si el usuario pone Min > Max o rango imposible, retornamos la media o rvs sin truncar con warning
            return rv.rvs(size=n_iter)

        # b. Generar uniformes en el rango [P_min, P_max]
        u = np.random.uniform(p_min, p_max, n_iter)
        
        # c. Invertir usando PPF (Percent Point Function)
        return rv.ppf(u)

# --- 4. HELPER PARA INPUTS DE UI ---

def render_dist_input(label, key, default_mode, default_min, default_max):
    """
    Renderiza inputs y opciones de truncamiento.
    """
    with st.expander(f"Configuración: {label}", expanded=False):
        options = ['BetaPERT', 'Lognormal', 'Normal', 'Triangular', 'Weibull', 'Gamma', 'Determinístico']
        # Seleccionar índice por defecto inteligente
        idx = 0 # PERT
        if "Declinación" in label: idx = 3 # Triangular o Normal
        
        dist = st.selectbox(f"Tipo de Distribución ({label})", options, index=idx, key=f"d_{key}")
        
        p = {}
        c1, c2, c3 = st.columns(3)
        
        # Inputs de Parámetros
        if dist == 'BetaPERT':
            p['min'] = c1.number_input("Mínimo", value=float(default_min), key=f"mn_{key}")
            p['mode'] = c2.number_input("Moda (Más Probable)", value=float(default_mode), key=f"md_{key}")
            p['max'] = c3.number_input("Máximo", value=float(default_max), key=f"mx_{key}")
        
        elif dist == 'Triangular':
            p['min'] = c1.number_input("Mínimo", value=float(default_min), key=f"tm_{key}")
            p['mode'] = c2.number_input("Moda", value=float(default_mode), key=f"tmd_{key}")
            p['max'] = c3.number_input("Máximo", value=float(default_max), key=f"tmx_{key}")
        
        elif dist == 'Normal':
            p['mean'] = c1.number_input("Media", value=float(default_mode), key=f"nm_{key}")
            delta = (default_max - default_min) / 6.0
            p['std'] = c2.number_input("Desv. Estándar", value=float(delta), key=f"ns_{key}")
        
        elif dist == 'Lognormal':
            p['mean'] = c1.number_input("Media", value=float(default_mode), key=f"lm_{key}")
            delta = (default_max - default_min) / 4.0
            p['std'] = c2.number_input("Desv. Estándar", value=float(delta), key=f"ls_{key}")
            
        elif dist == 'Weibull':
            p['shape'] = c1.number_input("Forma (k)", value=1.5, min_value=0.1, key=f"ws_{key}")
            p['scale'] = c2.number_input("Escala (λ)", value=float(default_mode), key=f"wsc_{key}")
            
        elif dist == 'Gamma':
            p['shape'] = c1.number_input("Forma (k)", value=2.0, min_value=0.1, key=f"gs_{key}")
            p['scale'] = c2.number_input("Escala (θ)", value=float(default_mode)/2, key=f"gsc_{key}")
            
        elif dist == 'Determinístico':
            p['value'] = c1.number_input("Valor Único", value=float(default_mode), key=f"dt_{key}")

        # --- SECCIÓN DE TRUNCAMIENTO ---
        # Solo para distribuciones que lo necesitan (Las abiertas)
        if dist in ['Normal', 'Lognormal', 'Weibull', 'Gamma']:
            st.markdown("---")
            st.markdown("**Límites de Truncamiento (Opcional)**")
            st.caption("Defina valores mínimos/máximos físicos que la variable no puede exceder.")
            
            use_trunc = st.checkbox(f"Aplicar Truncamiento a {label}", key=f"chk_{key}")
            if use_trunc:
                ct1, ct2 = st.columns(2)
                # Valores por defecto amplios para no cortar si el usuario no los toca
                t_min = ct1.number_input("Límite Mínimo (Trunc)", value=0.0, key=f"tmn_{key}")
                t_max = ct2.number_input("Límite Máximo (Trunc)", value=float(default_max)*5, key=f"tmx_{key}")
                p['trunc_min'] = t_min
                p['trunc_max'] = t_max
            else:
                p['trunc_min'] = -np.inf
                p['trunc_max'] = np.inf

    return {'dist': dist, 'params': p}

# --- 5. ECUACIONES FÍSICAS ---

def ipr_oil_darcy(k, h, delta_p, mu, Bo, re, rw, S):
    numerator = k * h * delta_p
    denominator = 141.2 * Bo * mu * (np.log(re/rw) + S)
    q = numerator / denominator
    return np.maximum(q, 0)

def ipr_oil_vogel(qmax, Pr_ref, delta_p):
    pwf = np.maximum(Pr_ref - delta_p, 0) 
    ratio = pwf / Pr_ref
    q = qmax * (1 - 0.2 * ratio - 0.8 * (ratio**2))
    return np.maximum(q, 0)

def ipr_gas_backpressure(C, Pr_ref, delta_p, n):
    pwf = np.maximum(Pr_ref - delta_p, 0)
    term = (Pr_ref**2 - pwf**2)
    q = C * (term**n)
    return q

def arps_forecast(t_array, qi_vec, di_vec, b_vec):
    di_m = di_vec / 12.0
    qi = qi_vec[:, np.newaxis]
    b = b_vec[:, np.newaxis]
    di = di_m[:, np.newaxis]
    t = t_array[np.newaxis, :]
    
    is_hyp = b > 0.001
    
    term_hyp = (1 + b * di * t)
    q_hyp = qi / (term_hyp ** (1.0 / np.maximum(b, 1e-9)))
    q_exp = qi * np.exp(-di * t)
    
    return np.where(is_hyp, q_hyp, q_exp)

# --- 6. APLICACIÓN PRINCIPAL ---

def main():
    with st.sidebar:
        try:
            st.image("mi-logo.png", use_container_width=True)
        except:
            st.warning("⚠️ Logo no cargado") 
        
        st.title("Configuración Global")
        fluid_type = st.radio("Tipo de Fluido", ["Aceite", "Gas"])
        n_iters = st.selectbox("Iteraciones Montecarlo", [1000, 5000, 10000], index=2)
    
    st.title("Reliarisk FlowCast")
    st.markdown("Plataforma Probabilística con **Truncamiento de Distribuciones**")

    tab1, tab2 = st.tabs(["🔹 Módulo I: Prod. Inicial (Afluencia)", "🔹 Módulo II: Pronóstico (Declinación)"])

    # --- MÓDULO I ---
    with tab1:
        st.header("Módulo I: Cálculo de Afluencia ($q_i$)")
        col_m1_1, col_m1_2 = st.columns([1, 1.5])
        
        with col_m1_1:
            st.info("Configure las variables y sus límites físicos (truncamiento).")
            
            if fluid_type == "Aceite":
                model_ipr = st.selectbox("Modelo IPR", ["Darcy (Flujo Radial)", "Vogel (Saturado)"])
            else:
                model_ipr = st.selectbox("Modelo IPR", ["Back Pressure (C & n)"])
            
            inputs_m1 = {}
            
            if fluid_type == "Aceite" and model_ipr == "Darcy (Flujo Radial)":
                inputs_m1['delta_p'] = render_dist_input("Drawdown (Pr - Pwf) [psi]", "dp_d", 500, 100, 1000)
                inputs_m1['k'] = render_dist_input("Permeabilidad k (mD)", "k", 50, 10, 100)
                inputs_m1['h'] = render_dist_input("Espesor h (ft)", "h", 100, 50, 150)
                inputs_m1['mu'] = render_dist_input("Viscosidad (cp)", "mu", 1.5, 1.0, 2.0)
                inputs_m1['Bo'] = render_dist_input("Factor Vol. Bo", "bo", 1.2, 1.1, 1.3)
                inputs_m1['S'] = render_dist_input("Daño (Skin)", "s", 0, -2, 5)
                re = st.number_input("Radio de Drene re (ft)", value=1000.0)
                rw = st.number_input("Radio del Pozo rw (ft)", value=0.328)

            elif fluid_type == "Aceite" and model_ipr == "Vogel (Saturado)":
                pr_ref = st.number_input("Presión Yacimiento Referencia (psi)", value=3000.0)
                inputs_m1['delta_p'] = render_dist_input("Drawdown (Pr - Pwf) [psi]", "dp_v", 500, 100, 1000)
                inputs_m1['qmax'] = render_dist_input("Qmax (AOF) bbl/d", "qmax", 5000, 3000, 8000)
                inputs_m1['Pr_ref'] = {'dist': 'Determinístico', 'params': {'value': pr_ref}}
            
            elif fluid_type == "Gas":
                pr_ref = st.number_input("Presión Yacimiento Referencia (psi)", value=3000.0)
                inputs_m1['delta_p'] = render_dist_input("Drawdown (Pr - Pwf) [psi]", "dp_g", 500, 100, 1000)
                inputs_m1['C'] = render_dist_input("Coeficiente C", "c_gas", 0.1, 0.01, 0.5)
                inputs_m1['n'] = render_dist_input("Exponente Turbulencia n", "n_gas", 0.8, 0.5, 1.0)
                inputs_m1['Pr_ref'] = {'dist': 'Determinístico', 'params': {'value': pr_ref}}

            btn_calc_m1 = st.button("Calcular $q_i$", key="btn_m1")

        with col_m1_2:
            if btn_calc_m1:
                # Generación de Muestras (Con Truncamiento)
                samples = {k: generate_samples(v['dist'], v['params'], n_iters) for k, v in inputs_m1.items()}
                
                if fluid_type == "Aceite" and model_ipr == "Darcy (Flujo Radial)":
                    qi_result = ipr_oil_darcy(samples['k'], samples['h'], samples['delta_p'], 
                                             samples['mu'], samples['Bo'], re, rw, samples['S'])
                    unit = "bbl/d"
                elif fluid_type == "Aceite" and model_ipr == "Vogel (Saturado)":
                    qi_result = ipr_oil_vogel(samples['qmax'], samples['Pr_ref'], samples['delta_p'])
                    unit = "bbl/d"
                elif fluid_type == "Gas":
                    qi_result = ipr_gas_backpressure(samples['C'], samples['Pr_ref'], samples['delta_p'], samples['n'])
                    unit = "MMPCD"

                st.session_state['qi_distribution'] = qi_result
                st.session_state['qi_unit'] = unit
                st.session_state['run_m1'] = True
                
                stats_qi = {
                    'P90': np.percentile(qi_result, 10),
                    'P50': np.percentile(qi_result, 50),
                    'P10': np.percentile(qi_result, 90)
                }
                
                st.success("✅ Afluencia Calculada")
                c1, c2, c3 = st.columns(3)
                c1.metric("P90", f"{stats_qi['P90']:.1f} {unit}")
                c2.metric("P50", f"{stats_qi['P50']:.1f} {unit}")
                c3.metric("P10", f"{stats_qi['P10']:.1f} {unit}")
                
                fig_hist = px.histogram(qi_result, nbins=50, title=f"Distribución $q_i$ ({unit})",
                                       color_discrete_sequence=['#004B87'])
                st.plotly_chart(fig_hist, use_container_width=True)

    # --- MÓDULO II ---
    with tab2:
        st.header("Módulo II: Pronóstico (Arps)")
        has_m1_data = st.session_state.get('run_m1', False)
        
        col_m2_1, col_m2_2 = st.columns([1, 1.5])
        
        with col_m2_1:
            st.subheader("Configuración")
            use_m1 = False
            if has_m1_data:
                st.success(f"Datos de Módulo I disponibles")
                use_m1 = st.checkbox("Usar $q_i$ del Módulo I", value=True)
            
            if not use_m1:
                input_qi_manual = render_dist_input("Gasto Inicial Qi", "qi_man", 1000, 500, 1500)

            input_di = render_dist_input("Declinación Inicial Di (Anual %)", "di", 0.20, 0.10, 0.40)
            input_b = render_dist_input("Exponente b", "b", 0.4, 0.0, 0.9)
            
            years = st.slider("Años", 1, 30, 20)
            qa_limit = st.number_input("Gasto Abandono", value=10.0)
            
            btn_calc_m2 = st.button("Ejecutar Pronóstico", key="btn_m2")

        with col_m2_2:
            if btn_calc_m2:
                if use_m1:
                    qi_vec = st.session_state['qi_distribution']
                    if len(qi_vec) != n_iters: qi_vec = np.random.choice(qi_vec, n_iters)
                else:
                    qi_vec = generate_samples(input_qi_manual['dist'], input_qi_manual['params'], n_iters)
                
                di_vec = generate_samples(input_di['dist'], input_di['params'], n_iters)
                b_vec = generate_samples(input_b['dist'], input_b['params'], n_iters)
                
                months = np.arange(0, years * 12 + 1)
                q_profiles = arps_forecast(months, qi_vec, di_vec, b_vec)
                q_profiles = np.where(q_profiles < qa_limit, 0, q_profiles)
                
                np_profiles = np.cumsum(q_profiles, axis=1) * 30.4167
                eur_vec = np_profiles[:, -1] / 1e6
                
                # Plots
                p10 = np.percentile(q_profiles, 90, axis=0)
                p50 = np.percentile(q_profiles, 50, axis=0)
                p90 = np.percentile(q_profiles, 10, axis=0)
                
                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(x=np.concatenate([months, months[::-1]]), y=np.concatenate([p90, p10[::-1]]), 
                                           fill='toself', fillcolor='rgba(0,75,135,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Incertidumbre'))
                fig_q.add_trace(go.Scatter(x=months, y=p50, name='P50', line=dict(color='#004B87')))
                fig_q.update_layout(title="Pronóstico Estocástico", yaxis_type="log", height=400)
                st.plotly_chart(fig_q, use_container_width=True)
                
                # EUR Metrics
                eur_p50 = np.percentile(eur_vec, 50)
                st.metric("Reservas Recuperables (EUR) - P50", f"{eur_p50:.2f} MM")

                # Sensitivity
                df_sens = pd.DataFrame({'Qi': qi_vec, 'Di': di_vec, 'b': b_vec, 'EUR': eur_vec})
                corr = df_sens.corr(method='spearman')['EUR'].drop('EUR').sort_values()
                fig_torn = px.bar(corr, orientation='h', title="Sensibilidad (Tornado)", color=corr)
                st.plotly_chart(fig_torn, use_container_width=True)

if __name__ == "__main__":
    main()