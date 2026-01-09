import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Reliarisk OHIP",
    page_icon="🛢️",
    layout="wide"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #004B87;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ESTADÍSTICAS ---

def generate_distribution(dist_type, params, n_iter):
    """Genera muestras aleatorias basadas en la distribución seleccionada."""
    if dist_type == 'Normal':
        return np.random.normal(params['mean'], params['std'], n_iter)
    elif dist_type == 'Lognormal':
        # Conversión básica para lognormal
        sigma = np.sqrt(np.log(1 + (params['std']/params['mean'])**2))
        mu = np.log(params['mean']) - 0.5 * sigma**2
        return np.random.lognormal(mu, sigma, n_iter)
    elif dist_type == 'Triangular':
        return np.random.triangular(params['min'], params['mode'], params['max'], n_iter)
    elif dist_type == 'Uniforme':
        return np.random.uniform(params['min'], params['max'], n_iter)
    elif dist_type == 'BetaPERT':
        # Aproximación PERT usando distribución Beta
        min_val, mode_val, max_val = params['min'], params['mode'], params['max']
        alpha = (4 * mode_val + max_val - 5 * min_val) / (max_val - min_val)
        beta = (5 * max_val - min_val - 4 * mode_val) / (max_val - min_val)
        return min_val + (max_val - min_val) * np.random.beta(alpha, beta, n_iter)
    elif dist_type == 'Determinístico':
        return np.full(n_iter, params['value'])
    return np.zeros(n_iter)

def apply_correlation(data_dict, corr_matrix):
    """
    Induce correlación de rango (Iman-Conover / Copula Gaussiana) a las muestras.
    data_dict: Diccionario {nombre_variable: array_muestras}
    corr_matrix: DataFrame de correlación
    """
    df = pd.DataFrame(data_dict)
    n_samples = df.shape[0]
    n_vars = df.shape[1]
    
    # 1. Generar matriz de puntuaciones normales con la correlación deseada
    mu = np.zeros(n_vars)
    # Asegurar que la matriz sea definida positiva (pequeño truco numérico si es necesario)
    try:
        L = np.linalg.cholesky(corr_matrix.values)
    except np.linalg.LinAlgError:
        # Fallback simple si la matriz no es positiva definida
        st.warning("Matriz de correlación no es positiva definida. Se ignoraron las correlaciones.")
        return df

    uncorrelated_normals = np.random.normal(0, 1, size=(n_samples, n_vars))
    correlated_normals = np.dot(uncorrelated_normals, L.T)
    
    # 2. Reordenar las muestras originales para que coincidan con el rango de las normales correlacionadas
    df_correlated = df.copy()
    cols = df.columns
    
    for i, col in enumerate(cols):
        # Obtener rangos de las normales correlacionadas
        rank_structure = np.argsort(np.argsort(correlated_normals[:, i]))
        # Ordenar los datos originales
        sorted_original = np.sort(df[col].values)
        # Aplicar la estructura de rangos
        df_correlated[col] = sorted_original[rank_structure]
        
    return df_correlated

# --- INTERFAZ DE USUARIO ---

# 1. Sidebar y Logo
with st.sidebar:
    try:
        st.image("mi_logo.png", use_container_width=True)
    except:
        st.warning("Archivo 'mi_logo.png' no encontrado. Cargue el logo en el directorio.")
    
    st.title("Configuración de Simulación")
    n_iterations = st.number_input("Iteraciones (Montecarlo)", value=10000, min_value=1000, step=1000)
    fluid_type = st.selectbox("Tipo de Fluido", ["Aceite", "Gas Seco"])

# 2. Entrada de Datos (Variables)
st.title("Reliarisk OHIP: Estimación Probabilística de Reservas")
st.markdown("---")

col1, col2 = st.columns([1, 2])

input_data = {}

with col1:
    st.subheader("Parámetros del Yacimiento")
    
    # Definición dinámica de variables según el fluido (Guía Plantilla No 1)
    # Aceite: N = 6.29 * A * h * phi * (1-Sw) / Bo [cite: 585]
    # Gas: G = 43560 * A * h * phi * (1-Sw) / Bg [cite: 690]
    
    variables = [
        {"key": "Area", "label": "Área (Km²)", "default_dist": "BetaPERT"},
        {"key": "Thickness", "label": "Espesor Neto (m)", "default_dist": "BetaPERT"},
        {"key": "Porosity", "label": "Porosidad (fracción)", "default_dist": "Normal"},
        {"key": "Sw", "label": "Saturación de Agua (fracción)", "default_dist": "Normal"},
        {"key": "FVF", "label": "Factor Vol. (Bo/Bg)", "default_dist": "Triangular"},
        {"key": "RF", "label": "Factor de Recuperación", "default_dist": "Uniforme"}
    ]
    
    params_store = {}

    for var in variables:
        with st.expander(f"Configurar {var['label']}", expanded=False):
            dist = st.selectbox(f"Distribución - {var['label']}", 
                                ['BetaPERT', 'Normal', 'Lognormal', 'Triangular', 'Uniforme', 'Determinístico'],
                                key=f"dist_{var['key']}", index=0)
            
            p = {}
            if dist == 'BetaPERT':
                p['min'] = st.number_input(f"Mínimo ({var['key']})", value=0.0, format="%.4f")
                p['mode'] = st.number_input(f"Más Probable ({var['key']})", value=0.0, format="%.4f")
                p['max'] = st.number_input(f"Máximo ({var['key']})", value=0.0, format="%.4f")
            elif dist == 'Normal':
                p['mean'] = st.number_input(f"Media ({var['key']})", value=0.0, format="%.4f")
                p['std'] = st.number_input(f"Desv. Estándar ({var['key']})", value=0.0, format="%.4f")
            elif dist == 'Lognormal':
                p['mean'] = st.number_input(f"Media ({var['key']})", value=0.0, format="%.4f")
                p['std'] = st.number_input(f"Desv. Estándar ({var['key']})", value=0.0, format="%.4f")
            elif dist == 'Triangular':
                p['min'] = st.number_input(f"Mínimo ({var['key']})", value=0.0, format="%.4f")
                p['mode'] = st.number_input(f"Moda ({var['key']})", value=0.0, format="%.4f")
                p['max'] = st.number_input(f"Máximo ({var['key']})", value=0.0, format="%.4f")
            elif dist == 'Uniforme':
                p['min'] = st.number_input(f"Mínimo ({var['key']})", value=0.0, format="%.4f")
                p['max'] = st.number_input(f"Máximo ({var['key']})", value=0.0, format="%.4f")
            elif dist == 'Determinístico':
                p['value'] = st.number_input(f"Valor ({var['key']})", value=0.0, format="%.4f")
            
            params_store[var['key']] = {'dist': dist, 'params': p}

    # 3. Matriz de Correlación [cite: 1453]
    st.subheader("Correlaciones")
    st.info("Defina la correlación (Spearman) entre Porosidad y Saturación de Agua (-1 a 1).")
    corr_phi_sw = st.slider("Correlación Porosidad - Sw", min_value=-1.0, max_value=1.0, value=-0.5, step=0.1)


# --- LÓGICA DE CÁLCULO Y VISUALIZACIÓN ---

with col2:
    if st.button("EJECUTAR SIMULACIÓN"):
        # 1. Generación de Muestras Independientes
        data = {}
        for var in variables:
            key = var['key']
            config = params_store[key]
            # Validar que los parámetros no sean cero (simple check)
            data[key] = generate_distribution(config['dist'], config['params'], n_iterations)

        # 2. Aplicar Correlaciones
        # Creamos una matriz base identidad
        vars_list = [v['key'] for v in variables]
        corr_df = pd.DataFrame(np.eye(len(vars_list)), index=vars_list, columns=vars_list)
        
        # Aplicar correlación específica definida por el usuario
        idx_phi = vars_list.index("Porosity")
        idx_sw = vars_list.index("Sw")
        corr_df.iloc[idx_phi, idx_sw] = corr_phi_sw
        corr_df.iloc[idx_sw, idx_phi] = corr_phi_sw
        
        sim_df = apply_correlation(data, corr_df)
        
        # 3. Cálculo Volumétrico [cite: 585, 690]
        # Ecuaciones según Guía Plantilla No 1
        if fluid_type == "Aceite":
            # N = 6.29 * A * h * phi * (1-Sw) / Boi (Eq 8)
            # A en Km2, h en m, Resultado en MMbls
            vol_original = (6.29 * sim_df['Area'] * sim_df['Thickness'] * sim_df['Porosity'] * (1 - sim_df['Sw'])) / sim_df['FVF']
            unit_label = "MMbls"
            product_label = "Aceite Original en Sitio (OOIP)"
        else:
            # G = 43560 * A * h * phi * (1-Sw) / Bgi (Eq 10 - ajustada para unidades)
            # Nota: La ecuación 10 usa Acres y Pies. La guía mezcla unidades.
            # Asumiremos la entrada consistente en Km2 y metros y convertiremos internamente
            # si el usuario sigue el input de la guía (Km2, m).
            # Para consistencia con la guía Eq 10 (que pide Acres y Pies), hacemos conversión:
            area_acres = sim_df['Area'] * 247.105
            h_feet = sim_df['Thickness'] * 3.28084
            vol_original = (43560 * area_acres * h_feet * sim_df['Porosity'] * (1 - sim_df['Sw'])) / sim_df['FVF']
            # Resultado en Pies Cúbicos. Convertir a BCF (Billions) o MMCF
            vol_original = vol_original / 1e9 # BCF
            unit_label = "BCF"
            product_label = "Gas Original en Sitio (OGIP)"

        # Reservas Recuperables [cite: 651]
        reserves = vol_original * sim_df['RF']

        # 4. Cálculo de Percentiles (P90, P50, P10) [cite: 684]
        # P90 = 90% probabilidad de exceder (valor bajo conservador)
        # P10 = 10% probabilidad de exceder (valor alto optimista)
        stats_res = {
            'P90': np.percentile(reserves, 10),
            'P50': np.percentile(reserves, 50),
            'P10': np.percentile(reserves, 90),
            'Mean': np.mean(reserves)
        }

        # --- DASHBOARD DE RESULTADOS [cite: 2007] ---
        st.subheader("Resultados de Reservas Recuperables")
        
        # Tarjetas de Métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P90 (Probado)", f"{stats_res['P90']:,.2f} {unit_label}")
        m2.metric("P50 (Mediana)", f"{stats_res['P50']:,.2f} {unit_label}")
        m3.metric("P10 (Posible)", f"{stats_res['P10']:,.2f} {unit_label}")
        m4.metric("Media", f"{stats_res['Mean']:,.2f} {unit_label}")

        # Gráfico 1: Histograma y Curva de Excedencia (Probabilidad) 
        fig_hist = go.Figure()
        
        # Histograma
        fig_hist.add_trace(go.Histogram(
            x=reserves, 
            name='Frecuencia',
            nbinsx=50,
            opacity=0.75,
            marker_color='#004B87',
            yaxis='y'
        ))

        # Curva de Excedencia (Eje Y secundario)
        sorted_res = np.sort(reserves)
        exceedance = 1.0 - np.arange(1, len(sorted_res) + 1) / len(sorted_res)
        
        fig_hist.add_trace(go.Scatter(
            x=sorted_res, 
            y=exceedance * 100, 
            name='Prob. Excedencia (%)',
            mode='lines',
            line=dict(color='#FF6347', width=3),
            yaxis='y2'
        ))

        # Líneas Verticales P10, P50, P90
        for p_label, p_val in stats_res.items():
            if p_label != 'Mean':
                fig_hist.add_vline(x=p_val, line_dash="dash", annotation_text=p_label)

        fig_hist.update_layout(
            title=f"Distribución de Reservas y Curva de Excedencia",
            xaxis_title=f"Reservas ({unit_label})",
            yaxis=dict(title="Frecuencia"),
            yaxis2=dict(title="Probabilidad Excedencia (%)", overlaying='y', side='right', range=[0, 100]),
            template="plotly_white",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Gráfico 2: Análisis de Sensibilidad (Tornado Chart) [cite: 2017]
        # Calculamos la correlación de rango de cada input con el resultado final
        sensitivity = {}
        for col in sim_df.columns:
            corr, _ = stats.spearmanr(sim_df[col], reserves)
            sensitivity[col] = corr
        
        sens_df = pd.DataFrame.from_dict(sensitivity, orient='index', columns=['Correlación'])
        sens_df = sens_df.sort_values(by='Correlación', key=abs, ascending=True)

        fig_tornado = px.bar(
            sens_df, 
            x='Correlación', 
            y=sens_df.index, 
            orientation='h',
            title="Diagrama de Tornado (Sensibilidad por Correlación de Rango)",
            color='Correlación',
            color_continuous_scale='RdBu'
        )
        fig_tornado.update_layout(xaxis_range=[-1, 1])
        st.plotly_chart(fig_tornado, use_container_width=True)

        # Exportar Datos
        csv = pd.DataFrame(reserves, columns=['Reservas_Recuperables']).to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar Resultados CSV",
            csv,
            "resultados_reliarisk.csv",
            "text/csv",
            key='download-csv'
        )