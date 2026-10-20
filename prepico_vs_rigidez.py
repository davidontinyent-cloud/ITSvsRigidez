import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# ---------------------------------------------------------
# 1. CONFIGURACIÓN E INTERFAZ
# ---------------------------------------------------------
st.set_page_config(page_title="Análisis de Rigidez y Ideal CT", layout="wide")

st.title("Módulo de Análisis Específico: Rigidez vs Ideal CT")

st.markdown("Este dashboard se centra exclusivamente en el análisis de la relación entre la Rigidez obtenida mediante el ensayo normativo UNE-EN 12697-26 y el ensayo normativo de rotura a Tracción Indirecta UNE-EN 12697-22. Dentro de los parámetros que se obtienen de la curva fuerza-desplazamiento, el análisis se ha centrado en la fase prepico que representa el comportamiento antes de la aparición de la macrofisura.")

@st.cache_data
def cargar_datos():
    if not os.path.exists('datos.xlsx'):
        st.error("No se encontró el archivo 'datos.xlsx'. Asegúrate de que el archivo Excel esté en el mismo directorio.")
        st.stop()
        
    try:
        df = pd.read_excel('datos.xlsx', engine='openpyxl')
        df.columns = df.columns.astype(str).str.strip()
        df = df.replace('-', np.nan)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned_list = [val.strip() if isinstance(val, str) else val for val in df[col]]
                    df[col] = cleaned_list
                except Exception:
                    pass
        
        cols_numericas = ['Carga_pico', 'CT_Index', 'Gf', 'Gf_prepico', 'Gf_postpico', 'm75', 'l75', 'lpeak', 'Rigidez_20']
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Espesor' in df.columns and 'Diametro' in df.columns:
            df['Tension_Rotura'] = (df['Carga_pico'] * 1000) / (df['Espesor'] * df['Diametro'])
            
        if 'Mezcla' in df.columns:
            if 'RAP' in df.columns and 'Envejecimiento' in df.columns:
                df['Todas'] = df['Mezcla'] + " | RAP: " + df['RAP'].astype(str) + " | Env: " + df['Envejecimiento'].astype(str)
            else:
                df['Todas'] = df['Mezcla']
        else:
            df['Todas'] = df.iloc[:, 0]
            
        # NUEVO: Cálculo dinámico del Centroide Y desde datos_brutos.xlsx
        if os.path.exists('datos_brutos.xlsx'):
            try:
                df_brutos = pd.read_excel('datos_brutos.xlsx', header=None, engine='openpyxl')
                probetas_brutos = df_brutos.iloc[3, :].values
                centroides = []
                
                for i in range(0, len(probetas_brutos), 2):
                    probeta_val = probetas_brutos[i]
                    if pd.isna(probeta_val): continue
                    
                    x = pd.to_numeric(df_brutos.iloc[5:, i], errors='coerce').dropna().values
                    y = pd.to_numeric(df_brutos.iloc[5:, i+1], errors='coerce').dropna().values
                    
                    min_len = min(len(x), len(y))
                    x, y = x[:min_len], y[:min_len]
                    
                    if len(x) > 2:
                        idx_peak = np.argmax(y)
                        x_prepico = x[:idx_peak+1]
                        y_prepico = y[:idx_peak+1]
                        
                        # Cálculo de áreas e integrales mediante regla trapezoidal
                        area = np.trapezoid(y_prepico, x_prepico)
                        integral_y2 = np.trapezoid(0.5 * y_prepico**2, x_prepico)
                        
                        if area > 0:
                            y_c = integral_y2 / area
                            centroides.append({'Probeta': probeta_val, 'Centroide_Y': y_c})
                
                if centroides:
                    df_centroides = pd.DataFrame(centroides)
                    df['Probeta_str'] = df['Probeta'].astype(str)
                    df_centroides['Probeta_str'] = df_centroides['Probeta'].astype(str)
                    df = df.merge(df_centroides[['Probeta_str', 'Centroide_Y']], on='Probeta_str', how='left')
                    df = df.drop(columns=['Probeta_str'])
                    
            except Exception as e:
                st.warning(f"Aviso: No se pudo procesar datos_brutos.xlsx para el cálculo del centroide: {e}")
                
        return df
        
    except Exception as e:
        st.error(f"Hubo un error cargando el archivo de datos: {e}")
        st.stop()
    return None

df = cargar_datos()

if 'l75' in df.columns and 'm75' in df.columns:
    df['abs_m75'] = df['m75'].abs()
    df['Ratio_Flexibilidad'] = (df['l75'] / df['abs_m75'].replace(0, np.nan)) * 1e6

if 'Gf_prepico' in df.columns and 'Tension_Rotura' in df.columns and 'abs_m75' in df.columns:
    df['Tension_Rotura_SI'] = df['Tension_Rotura'] * 1e6
    df['Num_Fisuracion'] = df['Gf_prepico'] * (df['Tension_Rotura_SI'] / 3)
    df['Indice_Fisuracion'] = df['Num_Fisuracion'] / df['abs_m75'].replace(0, np.nan)
    df['TR_sobre_m75'] = df['Tension_Rotura_SI'] / df['abs_m75'].replace(0, np.nan)

st.markdown("---")

# ---------------------------------------------------------
# 2. PESTAÑAS DE NAVEGACIÓN
# ---------------------------------------------------------
tab_reg, tab_rigidez_tension = st.tabs([
    "Comparativa Rigidez",
    "Rigidez vs Tensión"
])

# --- PESTAÑA 1: COMPARATIVA RIGIDEZ ---
with tab_reg:
    st.header("1. Comparativa Rigidez")
    
    df_tab4 = df[df['Velocidad'] == 50].copy()
    
    mezclas_unicas = df_tab4['Mezcla'].dropna().unique().tolist()
    mezcla_elegida = st.selectbox(
        "Selección de mezcla:", 
        ['Todas'] + mezclas_unicas,
        help="Si eliges 'Todas', el modelo de regresión evaluará el comportamiento global."
    )
    
    if mezcla_elegida == 'Todas':
        df_est = df_tab4.copy()
    else:
        df_est = df_tab4[df_tab4['Mezcla'] == mezcla_elegida].copy()
        
    st.markdown("---")
    
    st.subheader("1.1. Influencia de la Pendiente Pre-Pico")
    st.markdown("Se denominan mxx_yy siendo xx el tanto por ciento de la carga pico a la que se calcula la pendiente y yy el rango en el que se calcula. P.e m30_10 es la pendiente al 30% de la carga pico calculada entre el 35% y el 45%")
    lista_pendientes = [
        'm30_10', 'm35_10', 'm40_10', 'm45_10', 'm50_10', 'm55_10', 'm60_10', 'm65_10', 'm70_10', 
        'm30_20', 'm35_20', 'm40_20', 'm45_20', 'm50_20', 'm55_20', 'm60_20', 'm65_20', 'm70_20'
    ]
    pendientes_validas = [p for p in lista_pendientes if p in df_est.columns]
    
    col_2A, col_2B = st.columns([1, 2])
    with col_2A:
        pend_selec = st.selectbox("Selecciona la pendiente a evaluar:", pendientes_validas)
        df_clean = df_est.dropna(subset=[pend_selec, 'Rigidez_20'])
        
        if len(df_clean) > 2:
            X_simple = sm.add_constant(df_clean[pend_selec])
            modelo_simple = sm.OLS(df_clean['Rigidez_20'], X_simple).fit()
            st.metric(label="Precisión del ajuste (R²)", value=f"{modelo_simple.rsquared:.4f}")
            p_val = modelo_simple.pvalues[pend_selec]
            if p_val < 0.05:
                st.success(f"**Significativo** (p-value: {p_val:.4f})")
            else:
                st.warning(f"**No significativo** (p-value: {p_val:.4f})")
        else:
            st.warning("Faltan datos para realizar la regresión.")
            
    with col_2B:
        if len(df_clean) > 2:
            fig_p, ax_p = plt.subplots(figsize=(9, 5))
            sns.scatterplot(data=df_clean, x=pend_selec, y='Rigidez_20', hue='Todas', palette='tab10', s=90, alpha=0.8, ax=ax_p)
            sns.regplot(data=df_clean, x=pend_selec, y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_p)
            if 'Probeta' in df_clean.columns:
                for idx, row in df_clean.iterrows():
                    if pd.notna(row['Probeta']):
                        ax_p.annotate(str(row['Probeta']), (row[pend_selec], row['Rigidez_20']), textcoords="offset points", xytext=(6, 6), ha='left', fontsize=8, alpha=0.8)
            ax_p.set_ylabel('Rigidez Real (MPa)')
            ax_p.grid(True, linestyle='--', alpha=0.5)
            ax_p.legend(title='Tipo de Probeta', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=9)
            fig_p.tight_layout()
            st.pyplot(fig_p)

    st.markdown("---")
    
    st.subheader("1.2. Parámetros de Rotura y Energéticos vs Rigidez")
    col_2C, col_2D = st.columns(2)
    with col_2C:
        st.markdown("**Carga Pico**")
        df_carga = df_est.dropna(subset=['Carga_pico', 'Rigidez_20'])
        if len(df_carga) > 2:
            X_carga = sm.add_constant(df_carga['Carga_pico'])
            mod_c = sm.OLS(df_carga['Rigidez_20'], X_carga).fit()
            fig_c, ax_c = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=df_carga, x='Carga_pico', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_c)
            sns.regplot(data=df_carga, x='Carga_pico', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_c)
            ax_c.set_xlabel('Carga Pico (kN)')
            ax_c.set_ylabel('Rigidez (MPa)')
            ax_c.grid(True, alpha=0.3)
            ax_c.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            st.caption(f"R²: {mod_c.rsquared:.4f} | p-value: {mod_c.pvalues['Carga_pico']:.4f}")
            
    with col_2D:
        st.markdown("**Tensión de Rotura**")
        if 'Tension_Rotura' in df_est.columns:
            df_tens = df_est.dropna(subset=['Tension_Rotura', 'Rigidez_20'])
            if len(df_tens) > 2:
                X_tens = sm.add_constant(df_tens['Tension_Rotura'])
                mod_t = sm.OLS(df_tens['Rigidez_20'], X_tens).fit()
                fig_t, ax_t = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_t)
                sns.regplot(data=df_tens, x='Tension_Rotura', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_t)
                ax_t.set_xlabel('Tensión Rotura (MPa)')
                ax_t.set_ylabel('Rigidez (MPa)')
                ax_t.grid(True, alpha=0.3)
                ax_t.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                fig_t.tight_layout()
                st.pyplot(fig_t)
                st.caption(f"R²: {mod_t.rsquared:.4f} | p-value: {mod_t.pvalues['Tension_Rotura']:.4f}")

    # NUEVO: Gráficos adicionales de regresión para Energía y Centroide
    col_2E, col_2F = st.columns(2)
    with col_2E:
        st.markdown("**Energía Pre-Pico ($G_{f, prepico}$)**")
        if 'Gf_prepico' in df_est.columns:
            df_gf = df_est.dropna(subset=['Gf_prepico', 'Rigidez_20'])
            if len(df_gf) > 2:
                X_gf = sm.add_constant(df_gf['Gf_prepico'])
                mod_gf = sm.OLS(df_gf['Rigidez_20'], X_gf).fit()
                fig_gf, ax_gf = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_gf, x='Gf_prepico', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_gf)
                sns.regplot(data=df_gf, x='Gf_prepico', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_gf)
                ax_gf.set_xlabel('Energía Pre-Pico (J/m²)')
                ax_gf.set_ylabel('Rigidez (MPa)')
                ax_gf.grid(True, alpha=0.3)
                ax_gf.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                fig_gf.tight_layout()
                st.pyplot(fig_gf)
                st.caption(f"R²: {mod_gf.rsquared:.4f} | p-value: {mod_gf.pvalues['Gf_prepico']:.4f}")

    with col_2F:
        st.markdown("**Centroide Vertical del Área Pre-Pico ($Y_c$)**")
        if 'Centroide_Y' in df_est.columns:
            df_yc = df_est.dropna(subset=['Centroide_Y', 'Rigidez_20'])
            if len(df_yc) > 2:
                X_yc = sm.add_constant(df_yc['Centroide_Y'])
                mod_yc = sm.OLS(df_yc['Rigidez_20'], X_yc).fit()
                fig_yc, ax_yc = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_yc, x='Centroide_Y', y='Rigidez_20', hue='Todas', palette='tab10', s=80, alpha=0.8, ax=ax_yc)
                sns.regplot(data=df_yc, x='Centroide_Y', y='Rigidez_20', scatter=False, color='black', line_kws={'linestyle': '--', 'alpha':0.6}, ax=ax_yc)
                ax_yc.set_xlabel('Centroide Vertical $Y_c$ (kN)')
                ax_yc.set_ylabel('Rigidez (MPa)')
                ax_yc.grid(True, alpha=0.3)
                ax_yc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                fig_yc.tight_layout()
                st.pyplot(fig_yc)
                st.caption(f"R²: {mod_yc.rsquared:.4f} | p-value: {mod_yc.pvalues['Centroide_Y']:.4f}")

    st.markdown("---")
    st.subheader("1.3. Matrices de Correlación")
    # NUEVO: Añadidos los parámetros al listado de variables macro
    macro_params = ['Rigidez_20', 'Carga_pico']
    if 'Tension_Rotura' in df_est.columns:
        macro_params.append('Tension_Rotura')
    if 'Gf_prepico' in df_est.columns:
        macro_params.append('Gf_prepico')
    if 'Centroide_Y' in df_est.columns:
        macro_params.append('Centroide_Y')
        
    if len(df_est) > 2 and pendientes_validas:
        st.markdown("**A. Correlación cruzada: Parámetros Macro (Rigidez, Rotura, Energía y Centroide) vs Pendientes pre-pico**")
        corr_full = df_est[macro_params + pendientes_validas].corr()
        corr_macro_pendientes = corr_full.loc[macro_params, macro_params + pendientes_validas]
        fig_heat1, ax_heat1 = plt.subplots(figsize=(20, len(macro_params) * 1.5))
        sns.heatmap(corr_macro_pendientes, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5, ax=ax_heat1, annot_kws={"size": 9})
        ax_heat1.tick_params(axis='y', rotation=0)
        st.pyplot(fig_heat1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**B. Correlación interna: Análisis de colinealidad entre Pendientes**")
        corr_pendientes = df_est[pendientes_validas].corr()
        mask = np.triu(np.ones_like(corr_pendientes, dtype=bool))
        fig_heat2, ax_heat2 = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr_pendientes, mask=mask, annot=True, cmap='YlGnBu', fmt=".4f", linewidths=0.5, ax=ax_heat2, annot_kws={"size": 8})
        st.pyplot(fig_heat2)


# --- PESTAÑA 2: RIGIDEZ VS TENSIÓN ---
with tab_rigidez_tension:
    st.header("2. Análisis de Interacción: Rigidez vs Tensión de Rotura")
    st.markdown("""
    Para evitar que la alta varianza estructural de una mezcla AC enmascare el comportamiento viscoelástico de una SMA, 
    la estrategia analítica fundamental es utilizar el tipo de mezcla como filtro primario y aplicar el concepto estadístico de 
    **"Múltiplos Pequeños"** (Small Multiples). Esto previene la Paradoja de Simpson, donde una tendencia global contradice el comportamiento real de los subgrupos.
    """)
    st.markdown("⚠️ *Análisis filtrado únicamente para ensayos a **50 mm/min**.*")
    
    if 'Tension_Rotura' in df.columns and 'Rigidez_20' in df.columns:
        df_rt = df[df['Velocidad'] == 50].dropna(subset=['Tension_Rotura', 'Rigidez_20', 'Mezcla', 'Envejecimiento', 'RAP']).copy()
        
        if not df_rt.empty:
            
            st.markdown("---")
            st.subheader("2.1. Relación Global: Tensión vs Rigidez (Todas las Probetas)")
            st.markdown("Diagrama de dispersión global con **líneas de tendencia independientes por mezcla** y una **Línea de Ajuste Total** (en negro discontinuo) que evalúa el coeficiente de correlación de todas las probetas en conjunto.")
            
            df_global = df_rt.copy()
            
            X_tot = sm.add_constant(df_global['Tension_Rotura'])
            mod_tot = sm.OLS(df_global['Rigidez_20'], X_tot).fit()
            m_tot = mod_tot.params.iloc[1] if len(mod_tot.params) > 1 else 0
            b_tot = mod_tot.params.iloc[0]
            r2_tot = mod_tot.rsquared

            fig_global_rt = px.scatter(
                df_global, x="Tension_Rotura", y="Rigidez_20", color="Mezcla",
                trendline="ols",
                hover_data=['Probeta', 'RAP', 'Envejecimiento'],
                labels={"Tension_Rotura": "Tensión de Rotura (MPa)", "Rigidez_20": "Rigidez a 20ºC (MPa)"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            x_min_tot, x_max_tot = df_global['Tension_Rotura'].min(), df_global['Tension_Rotura'].max()
            x_rango = np.array([x_min_tot, x_max_tot])
            y_rango = b_tot + m_tot * x_rango
            
            fig_global_rt.add_trace(go.Scatter(
                x=x_rango, y=y_rango,
                mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                name=f'Tendencia TOTAL (R² = {r2_tot:.3f})'
            ))
            
            fig_global_rt.update_traces(marker=dict(size=8, opacity=0.8))
            fig_global_rt.update_layout(height=500)
            st.plotly_chart(fig_global_rt, use_container_width=True)
            
            st.markdown("---")
            st.subheader("2.2. Análisis Aislado: Rigidez y Tensión de Rotura")
            st.markdown("""
            Panel independiente para cada mezcla cuantificando el impacto del envejecimiento y del RAP. 
            Permite ver si el RAP actúa puramente como rigidizador o aporta fuerza, y si la varianza (ancho de caja) se vuelve inestable en E2.
            """)
            
            var_aislada = st.radio("Selecciona la variable a diseccionar:", ["Tension_Rotura", "Rigidez_20"], horizontal=True)
            y_label_aislada = "Tensión de Rotura (MPa)" if var_aislada == "Tension_Rotura" else "Rigidez a 20ºC (MPa)"
            
            fig_box_rt = px.box(
                df_rt, x="Envejecimiento", y=var_aislada, color="RAP", facet_col="Mezcla",
                category_orders={"Envejecimiento": ["Control", "E1", "E2"]},
                labels={var_aislada: y_label_aislada},
                color_discrete_sequence=px.colors.qualitative.Set1,
                points="all"
            )
            
            df_medians_rt = df_rt.groupby(['Mezcla', 'Envejecimiento', 'RAP'])[var_aislada].median().reset_index()
            order_env = {'Control': 0, 'E1': 1, 'E2': 2}
            df_medians_rt['orden'] = df_medians_rt['Envejecimiento'].map(order_env)
            df_medians_rt = df_medians_rt.sort_values(['Mezcla', 'RAP', 'orden'])

            for i, mezcla in enumerate(df_rt['Mezcla'].unique()):
                df_mezcla = df_medians_rt[df_medians_rt['Mezcla'] == mezcla]
                for rap in df_mezcla['RAP'].unique():
                    df_rap = df_mezcla[df_mezcla['RAP'] == rap]
                    fig_box_rt.add_trace(go.Scatter(
                        x=df_rap['Envejecimiento'], y=df_rap[var_aislada],
                        mode='lines+markers',
                        line=dict(width=2, dash='dot', color='rgba(50,50,50,0.6)'),
                        marker=dict(size=6, color='rgba(50,50,50,0.8)', symbol='diamond'),
                        showlegend=False,
                        hoverinfo='skip',
                        xaxis=f'x{i+1}' if i > 0 else 'x',
                        yaxis=f'y{i+1}' if i > 0 else 'y'
                    ))

            fig_box_rt.update_yaxes(matches=None) 
            fig_box_rt.update_layout(height=500, boxmode="group")
            st.plotly_chart(fig_box_rt, use_container_width=True)
            
            st.markdown("---")
            st.subheader("2.3. Relación Bivariante: Trayectorias Cinemáticas por Mezcla")
            st.markdown("""
            Representación de Rigidez (X) frente a Tensión de Rotura (Y). Las líneas conectan los **centroides** de los datos 
            (`Control → E1 → E2`) trazando el vector de deterioro **separado por contenido de RAP**.
            *   **Vector a 45º:** La mezcla gana fuerza al mismo ritmo que se endurece.
            *   **Vector plano hacia la derecha:** La mezcla se rigidiza sin ganar resistencia (fragilización inminente).
            """)
            
            mezclas_rt = df_rt['Mezcla'].unique()
            cols_rt = st.columns(len(mezclas_rt))
            
            for i, mezcla in enumerate(mezclas_rt):
                with cols_rt[i]:
                    st.markdown(f"**Mezcla: {mezcla}**")
                    df_m = df_rt[df_rt['Mezcla'] == mezcla]
                    
                    rap_types = df_m['RAP'].dropna().unique()
                    palette = px.colors.qualitative.Dark2
                    color_map = {rap: palette[j % len(palette)] for j, rap in enumerate(rap_types)}
                    
                    fig_scatter_rt = px.scatter(
                        df_m, x="Rigidez_20", y="Tension_Rotura", color="RAP",
                        hover_data=['Probeta', 'Envejecimiento'],
                        labels={"Rigidez_20": "Rigidez (MPa)", "Tension_Rotura": "Tensión Rotura"},
                        color_discrete_map=color_map
                    )
                    
                    centroides = df_m.groupby(['RAP', 'Envejecimiento'])[['Rigidez_20', 'Tension_Rotura']].mean().reset_index()
                    orden_env = {'Control': 0, 'E1': 1, 'E2': 2}
                    centroides['orden'] = centroides['Envejecimiento'].map(orden_env)
                    
                    for rap_val in rap_types:
                        cent_subset = centroides[centroides['RAP'] == rap_val].sort_values('orden').dropna()
                        
                        if len(cent_subset) > 1:
                            fig_scatter_rt.add_trace(go.Scatter(
                                x=cent_subset['Rigidez_20'], y=cent_subset['Tension_Rotura'],
                                mode='lines+markers+text',
                                marker=dict(symbol='star', size=14, color=color_map[rap_val], line=dict(width=1, color='black')),
                                line=dict(color=color_map[rap_val], width=2, dash='dot'),
                                text=cent_subset['Envejecimiento'],
                                textposition="top center",
                                showlegend=False,
                                name=f"Trayectoria ({rap_val})"
                            ))
                    
                    fig_scatter_rt.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
                    if i > 0:
                        fig_scatter_rt.update_yaxes(title_text="")
                        
                    st.plotly_chart(fig_scatter_rt, use_container_width=True)
            
            st.caption("Puntos y trayectorias coloreadas por RAP. Las estrellas y líneas punteadas representan la migración del centroide de cada subgrupo.")

            st.markdown("---")
            st.subheader("2.4. Cuantificación Lineal (Coeficientes de Pearson)")
            st.markdown("Cálculo del coeficiente de correlación ($r$) entre Rigidez y Tensión aislando cada clúster de Mezcla y RAP.")
            
            res_corr = []
            for name, group in df_rt.groupby(['Mezcla', 'RAP']):
                if len(group) > 2:
                    corr = group['Rigidez_20'].corr(group['Tension_Rotura'])
                    res_corr.append({'Mezcla': name[0], 'RAP': name[1], 'Correlación de Pearson (r)': corr})
            
            if res_corr:
                df_corr = pd.DataFrame(res_corr)
                st.dataframe(df_corr.style.format({'Correlación de Pearson (r)': '{:.3f}'})
                                         .background_gradient(cmap='RdYlGn', subset=['Correlación de Pearson (r)'], vmin=-1, vmax=1))
            else:
                st.info("No hay suficientes datos por clúster para calcular correlaciones individuales.")
                
        else:
            st.warning("No hay suficientes datos válidos para este análisis.")
    else:
        st.error("Para poder utilizar esta pestaña es necesario que el archivo contenga las columnas 'Tension_Rotura' y 'Rigidez_20'.")