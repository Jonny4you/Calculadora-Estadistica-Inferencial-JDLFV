import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Calculadora Estadística",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCIÓN DE UTILIDAD: Z-score a partir del nivel de confianza ---
# Usamos cache para no recalcular este valor en cada recarga
@st.cache_data
def get_z_score(confianza):
    """Calcula el valor Z crítico para un nivel de confianza dado (dos colas)."""
    return stats.norm.ppf(1 - (1 - confianza) / 2)

# --- FUNCIÓN PRINCIPAL DE LA APP ---
def main():
    st.title("📊 Calculadora Estadística Avanzada")
    st.markdown("---")

    # --- NAVEGACIÓN PRINCIPAL (Sidebar) ---
    st.sidebar.title("Menú Principal")
    seccion = st.sidebar.radio(
        "Selecciona una Sección:",
        ["Medidas de Tendencia Central", "Inferencia Estadística"]
    )

    # Inicialización de session_state para persistir resultados y evitar KeyErrors
    if 'resultado_una' not in st.session_state: st.session_state['resultado_una'] = None
    if 'tipo_calculo_una' not in st.session_state: st.session_state['tipo_calculo_una'] = None
    if 'resultado_dos' not in st.session_state: st.session_state['resultado_dos'] = None
    if 'tipo_calculo_dos' not in st.session_state: st.session_state['tipo_calculo_dos'] = None


    # --- SECCIÓN 1: Medidas de Tendencia Central (Pestañas 1 y 2) ---
    if seccion == "Medidas de Tendencia Central":
        st.header("1️⃣ Medidas de Tendencia Central")

        tab_input, tab_resultados = st.tabs(["Datos de Entrada (1ra Pestaña)", "Resultados (2da Pestaña)"])

        with tab_input:
            st.subheader("Ingreso de Datos")
            datos_raw = st.text_area(
                "Ingresa tus datos separados por comas, espacios o saltos de línea (ej: 1, 5, 10, 15, 20)",
                height=150, key='tc_data'
            )
            calcular_tendencia = st.button("Calcular Medidas", key='calc_tc_btn')
            
            datos = None
            if datos_raw and calcular_tendencia:
                try:
                    datos_list = [float(x.strip()) for x in datos_raw.replace(',', ' ').split() if x.strip()]
                    datos = np.array(datos_list)
                    if len(datos) == 0:
                        st.warning("Por favor, ingresa datos válidos.")
                        datos = None
                except ValueError:
                    st.error("Error: Asegúrate de que los datos sean números válidos.")
                    datos = None

        with tab_resultados:
            st.subheader("Resultados del Análisis Descriptivo (2da Pestaña)")
            if calcular_tendencia and datos is not None and len(datos) > 0:
                # Cálculo de Medidas de Tendencia Central
                media = np.mean(datos)
                mediana = np.median(datos)
                
                # --- Lógica de la MODA CORREGIDA (Versión Definitiva) ---
                # stats.mode retorna un objeto ModeResult. Usamos keepdims=False para compatibilidad.
                moda_res = stats.mode(datos, keepdims=False)
                
                # Convertimos el resultado a un array de NumPy para asegurar que tenga la propiedad len()
                moda_array = np.atleast_1d(moda_res.mode)
                
                if len(moda_array) == 0 or np.all(np.isnan(moda_array)):
                    moda = "No hay moda (datos únicos o dispersos)"
                elif len(moda_array) == 1:
                    moda = f"{moda_array[0]:,.4f}"
                else:
                    # En caso de moda múltiple
                    moda_valores = ", ".join([f"{m:,.4f}" for m in moda_array])
                    moda = f"Múltiple: {moda_valores}"
                
                # Creación de la tabla de resultados
                df_resultados = pd.DataFrame({
                    "Medida": ["Media", "Mediana", "Moda", "Recuento (n)"],
                    "Valor": [f"{media:,.4f}", f"{mediana:,.4f}", moda, len(datos)]
                })
                st.dataframe(df_resultados.set_index('Medida'))
                
                st.markdown("---")

                # Extra Opcional: Histograma
                st.subheader("Extra Opcional: Histograma (Distribución de Frecuencias)")
                st.markdown("El histograma muestra visualmente la distribución de los datos y dónde se concentran las **Medidas de Tendencia Central**.")
                fig, ax = plt.subplots()
                ax.hist(datos, bins='auto', edgecolor='black', alpha=0.7)
                ax.set_title('Histograma de los Datos')
                ax.set_xlabel('Valor')
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
            elif calcular_tendencia:
                 st.warning("Por favor, ingresa datos válidos para calcular.")


    # --- SECCIÓN 2: Inferencia Estadística (Pestañas 3, 4, 5, 6) ---
    elif seccion == "Inferencia Estadística":
        st.header("2️⃣ Inferencia Estadística")

        # Pestañas para Una Población (3ra) y Dos Poblaciones (5ta)
        tabs_poblacion = st.tabs(["3️⃣ Una Población", "5️⃣ Dos Poblaciones"])

        # --- SUBSECCIÓN: UNA POBLACIÓN (3ra Pestaña) ---
        with tabs_poblacion[0]:
            st.subheader("Cálculos para Una Única Población (3ra Pestaña)")
            
            # Sub-pestañas para cálculo (3ra Pestaña) y resultados (4ta Pestaña)
            calc_una, result_una = st.tabs(["Calculadora (3ra Pestaña)", "Resultados (4ta Pestaña)"])

            with calc_una:
                opcion_una = st.selectbox(
                    "Selecciona el Cálculo:",
                    [
                        "Error Estándar de la Media",
                        "Intervalo de Confianza de la Media",
                        "Intervalo de Confianza de una Proporción",
                        "Cálculo de Z y T-student (Estadísticos)",
                        "Tamaño de Muestra por Media",
                        "Tamaño de Muestra por Proporción"
                    ], key='op_una'
                )
                
                st.info(f"Inputs para: **{opcion_una}**")
                
                # --- Lógica de inputs: Error Estándar de la Media ---
                if opcion_una == "Error Estándar de la Media":
                    # Mantenemos las variables locales referenciadas a las claves del session_state para que Streamlit sepa que las debe leer en la re-ejecución
                    s_ee = st.number_input("Desviación estándar muestral (s)", value=10.0, key='s_ee_input')
                    n_ee = st.number_input("Tamaño de la muestra (n)", min_value=1, value=30, key='n_ee_input')
                    
                    if st.button("Calcular Error Estándar", key='btn_ee_calc'):
                        st.session_state['tipo_calculo_una'] = 'EE_Media'
                        # Almacenamos directamente los valores leídos
                        st.session_state['s_ee'] = s_ee
                        st.session_state['n_ee'] = n_ee


                # --- Lógica de inputs: IC de la Media ---
                elif opcion_una == "Intervalo de Confianza de la Media":
                    # Usamos claves terminadas en '_input' o '_slider' para evitar conflictos con las claves de guardado.
                    media_m = st.number_input("Media muestral (x̄)", value=50.0, key='media_icm_input')
                    desv_std = st.number_input("Desviación estándar de la muestra (s)", value=10.0, key='desv_icm_input')
                    n = st.number_input("Tamaño de la muestra (n)", min_value=2, value=30, key='n_icm_input')
                    confianza = st.slider("Nivel de Confianza (%)", min_value=80, max_value=99, value=95, key='conf_icm_slider') / 100.0
                    
                    if st.button("Calcular Intervalo (Media)", key='btn_icm_calc'):
                        st.session_state['tipo_calculo_una'] = 'IC_Media'
                        # Almacenamos directamente los valores leídos
                        st.session_state['confianza_icm'] = confianza
                        st.session_state['n_icm'] = n
                        st.session_state['media_icm'] = media_m
                        st.session_state['desv_icm'] = desv_std

                
                # --- Lógica de inputs: IC de una Proporción ---
                elif opcion_una == "Intervalo de Confianza de una Proporción":
                    x_exitos = st.number_input("Número de Éxitos (x)", min_value=0, value=15, key='x_icp_input')
                    n_total = st.number_input("Tamaño de la muestra (n)", min_value=1, value=50, key='n_icp_input')
                    confianza_p = st.slider("Nivel de Confianza (%)", min_value=80, max_value=99, value=95, key='conf_icp_slider') / 100.0
                    
                    if st.button("Calcular Intervalo (Proporción)", key='btn_icp_calc'):
                        if x_exitos > n_total:
                            st.error("El número de éxitos (x) no puede ser mayor que el tamaño de la muestra (n).")
                        else:
                            st.session_state['tipo_calculo_una'] = 'IC_Prop'
                            st.session_state['confianza_icp'] = confianza_p
                            # Almacenamos directamente los valores leídos
                            st.session_state['x_icp'] = x_exitos 
                            st.session_state['n_icp'] = n_total

                # --- Lógica de inputs: Cálculo de Z y T-student ---
                elif opcion_una == "Cálculo de Z y T-student (Estadísticos)":
                    media_muestral = st.number_input("Media muestral (x̄)", value=50.0, key='media_zt')
                    media_hipotesis = st.number_input("Media hipotética (μ₀)", value=48.0, key='mu_zt')
                    desv_zt = st.number_input("Desviación estándar muestral (s)", value=5.0, key='s_zt')
                    n_zt = st.number_input("Tamaño de la muestra (n)", min_value=1, value=25, key='n_zt')
                    
                    metodo_zt = st.radio("Estadístico a calcular:", ["T-student (σ desconocida)", "Z (n grande o σ conocida)"], key='metodo_zt')

                    if st.button("Calcular Estadísticos"):
                        st.session_state['tipo_calculo_una'] = 'Z_T_Calc'
                        st.session_state['media_muestral_zt'] = media_muestral
                        st.session_state['media_hipotesis_zt'] = media_hipotesis
                        st.session_state['desv_zt'] = desv_zt
                        st.session_state['n_zt'] = n_zt
                        st.session_state['metodo_zt'] = metodo_zt
                        
                # --- Lógica de inputs: Tamaño de Muestra por Media ---
                elif opcion_una == "Tamaño de Muestra por Media":
                    confianza_tm = st.slider("Nivel de Confianza (%)", min_value=80, max_value=99, value=95, key='conf_tmm') / 100.0
                    desv_tm = st.number_input("Desviación estándar estimada (s o σ)", value=10.0, key='s_tmm')
                    margen_error_tm = st.number_input("Margen de Error deseado (E)", value=1.5, min_value=0.01, key='e_tmm')

                    if st.button("Calcular Tamaño de Muestra (Media)"):
                        st.session_state['tipo_calculo_una'] = 'TM_Media'
                        st.session_state['confianza_tmm'] = confianza_tm
                        st.session_state['desv_tmm'] = desv_tm
                        st.session_state['margen_error_tmm'] = margen_error_tm

                # --- Lógica de inputs: Tamaño de Muestra por Proporción ---
                elif opcion_una == "Tamaño de Muestra por Proporción":
                    confianza_tmp = st.slider("Nivel de Confianza (%)", min_value=80, max_value=99, value=95, key='conf_tmp') / 100.0
                    prop_estimada = st.number_input("Proporción estimada (p̂) [0-1]", value=0.5, min_value=0.01, max_value=0.99, key='p_tmp')
                    margen_error_tmp = st.number_input("Margen de Error deseado (E)", value=0.03, min_value=0.001, key='e_tmp')

                    if st.button("Calcular Tamaño de Muestra (Proporción)"):
                        st.session_state['tipo_calculo_una'] = 'TM_Prop'
                        st.session_state['confianza_tmp'] = confianza_tmp
                        st.session_state['prop_tmp'] = prop_estimada
                        st.session_state['margen_error_tmp'] = margen_error_tmp


            with result_una:
                st.subheader("Resultados de Una Población (4ta Pestaña)")
                tipo_calc = st.session_state['tipo_calculo_una']
                
                # --- Resultados: Error Estándar de la Media ---
                if tipo_calc == 'EE_Media':
                    s, n = st.session_state['s_ee'], st.session_state['n_ee']
                    ee = s / np.sqrt(n)
                    st.success(f"El Error Estándar (EE) es: **{ee:,.4f}**")
                    st.write(r"Cálculo: $EE = s / \sqrt{n} = {s:,.2f} / \sqrt{{{n}}} = {ee:,.4f}$")
                    
                    st.markdown("---")
                    st.subheader("Extra Opcional: Comportamiento del EE")
                    st.markdown("El **Error Estándar (EE)** mide qué tan lejos está la media muestral de la media poblacional. Es la desviación estándar de la distribución muestral de la media. **A mayor *n*, menor EE.**")
                
                # --- Resultados: IC de la Media ---
                elif tipo_calc == 'IC_Media':
                    media_m = st.session_state['media_icm']
                    desv_std = st.session_state['desv_icm']
                    n = st.session_state['n_icm']
                    confianza = st.session_state['confianza_icm']
                    alfa = 1 - confianza
                    
                    # Asumiendo T-student para IC de la media, ya que s es muestral
                    grados_libertad = n - 1
                    t_score = stats.t.ppf(1 - alfa/2, grados_libertad)
                    
                    error_estandar_ic = desv_std / np.sqrt(n)
                    margen_error = t_score * error_estandar_ic
                    
                    lim_inf = media_m - margen_error
                    lim_sup = media_m + margen_error
                    
                    st.success(f"El Intervalo de Confianza ({confianza*100}%) es: **[{lim_inf:,.4f}, {lim_sup:,.4f}]**")
                    st.write(f"**Estadístico de Prueba (T):** {t_score:,.4f} (gl={grados_libertad})")
                    st.write(f"**Margen de Error (E):** {margen_error:,.4f}")
                    
                    st.markdown("---")
                    st.subheader("Extra Opcional: Distribución muestral")
                    st.markdown("Para este cálculo se utilizó la distribución **T-student** (debido a que se usó la desviación estándar muestral *s*). A medida que $n$ crece, la T-student se aproxima a la Normal Z.")
                                    # --- Resultados: IC de una Proporción ---
                elif tipo_calc == 'IC_Prop':
                    x, n = st.session_state['x_icp'], st.session_state['n_icp']
                    confianza = st.session_state['confianza_icp']
                    
                    phat = x / n
                    z_score = get_z_score(confianza)
                    
                    ee_prop = np.sqrt(phat * (1 - phat) / n)
                    margen_error = z_score * ee_prop
                    
                    lim_inf = phat - margen_error
                    lim_sup = phat + margen_error
                    
                    st.success(f"El Intervalo de Confianza ({confianza*100}%) para la proporción es: **[{lim_inf:,.4f}, {lim_sup:,.4f}]**")
                    st.write(f"**Proporción muestral (p̂):** {phat:,.4f}")
                    st.write(f"**Estadístico de Prueba (Z):** {z_score:,.4f}")
                    st.write(f"**Margen de Error (E):** {margen_error:,.4f}")
                
                # --- Resultados: Cálculo de Z y T-student ---
                elif tipo_calc == 'Z_T_Calc':
                    x_bar = st.session_state['media_muestral_zt']
                    mu_0 = st.session_state['media_hipotesis_zt']
                    s = st.session_state['desv_zt']
                    n = st.session_state['n_zt']
                    metodo = st.session_state['metodo_zt']

                    ee = s / np.sqrt(n)
                    estadistico = (x_bar - mu_0) / ee
                    
                    if "T-student" in metodo:
                        gl = n - 1
                        st.success(f"Estadístico T-student Calculado: **{estadistico:,.4f}**")
                        st.write(f"Grados de Libertad (gl): {gl}")
                        st.write(r"Cálculo: $T = (\bar{x} - \mu_0) / (s / \sqrt{n}) = ({x_bar} - {mu_0}) / ({s} / \sqrt{{{n}}}) = {estadistico:,.4f}$")
                        
                    else: # Z
                        st.success(f"Estadístico Z Calculado: **{estadistico:,.4f}**")
                        st.write(r"Cálculo: $Z = (\bar{x} - \mu_0) / (s / \sqrt{n}) = ({x_bar} - {mu_0}) / ({s} / \sqrt{{{n}}}) = {estadistico:,.4f}$")
                        
                # --- Resultados: Tamaño de Muestra por Media ---
                elif tipo_calc == 'TM_Media':
                    conf = st.session_state['confianza_tmm']
                    s = st.session_state['desv_tmm']
                    E = st.session_state['margen_error_tmm']
                    
                    Z = get_z_score(conf)
                    n_calc = ((Z * s) / E)**2
                    n_redondeado = int(np.ceil(n_calc))
                    
                    st.success(f"Tamaño de Muestra Requerido (n): **{n_redondeado}**")
                    st.write(r"Cálculo: $n = (Z \cdot s / E)^2 = ({Z:,.4f} \cdot {s} / {E})^2 = {n_calc:,.2f} \implies {n_redondeado}$")
                    st.write(f"*(Se debe redondear al entero superior para asegurar el margen de error)*")

                # --- Resultados: Tamaño de Muestra por Proporción ---
                elif tipo_calc == 'TM_Prop':
                    conf = st.session_state['confianza_tmp']
                    p = st.session_state['prop_tmp']
                    E = st.session_state['margen_error_tmp']
                    
                    Z = get_z_score(conf)
                    n_calc = p * (1 - p) * ((Z / E)**2)
                    n_redondeado = int(np.ceil(n_calc))
                    
                    st.success(f"Tamaño de Muestra Requerido (n): **{n_redondeado}**")
                    st.write(r"Cálculo: $n = p(1-p) \cdot (Z/E)^2 = {p} \cdot {1-p} \cdot ({Z:,.4f} / {E})^2 = {n_calc:,.2f} \implies {n_redondeado}$")
                    st.write(f"*(Se debe redondear al entero superior para asegurar el margen de error)*")

                elif tipo_calc is None:
                    st.info("El resultado del cálculo de 'Una Población' aparecerá aquí.")

        # --- SUBSECCIÓN: DOS POBLACIONES (5ta Pestaña) ---
        with tabs_poblacion[1]:
            st.subheader("Cálculos para Dos Poblaciones (Comparación) (5ta Pestaña)")

            # Sub-pestañas para cálculo (5ta Pestaña) y resultados (6ta Pestaña)
            calc_dos, result_dos = st.tabs(["Calculadora (5ta Pestaña)", "Resultados (6ta Pestaña)"])

            with calc_dos:
                opcion_dos = st.selectbox(
                    "Selecciona el Cálculo:",
                    [
                        "Diferencia de Medias (Intervalo de Confianza)",
                        "Diferencia de Proporciones (Intervalo de Confianza)",
                        "Prueba de Hipótesis para Medias",
                        "Prueba de Hipótesis para Proporciones"
                    ], key='op_dos_select' # Clave única para selectbox
                )
                
                st.info(f"Inputs para: **{opcion_dos}**")
                
                # --- INPUTS CONDICIONALES ---

                if "Media" in opcion_dos:
                    # Usamos claves únicas para Medias (m1_dos, d1_dos, etc.)
                    st.markdown("#### Población 1")
                    media1 = st.number_input("Media muestral 1 (x̄₁)", value=60.0, key='m1_dos')
                    desv1 = st.number_input("Desviación estándar 1 (s₁)", value=8.0, key='d1_dos')
                    n1 = st.number_input("Tamaño de la muestra 1 (n₁)", min_value=2, value=40, key='n1_dos')

                    st.markdown("#### Población 2")
                    media2 = st.number_input("Media muestral 2 (x̄₂)", value=55.0, key='m2_dos')
                    desv2 = st.number_input("Desviación estándar 2 (s₂)", value=7.0, key='d2_dos')
                    n2 = st.number_input("Tamaño de la muestra 2 (n₂)", min_value=2, value=35, key='n2_dos')

                elif "Proporción" in opcion_dos:
                    # Usamos claves únicas para Proporciones (x1_dos, n1_dos_p, etc.)
                    st.markdown("#### Población 1")
                    x1 = st.number_input("Éxitos 1 (x₁)", min_value=0, value=25, key='x1_dos')
                    n1 = st.number_input("Tamaño de la muestra 1 (n₁)", min_value=1, value=50, key='n1_dos_p')

                    st.markdown("#### Población 2")
                    x2 = st.number_input("Éxitos 2 (x₂)", min_value=0, value=30, key='x2_dos')
                    n2 = st.number_input("Tamaño de la muestra 2 (n₂)", min_value=1, value=70, key='n2_dos_p')
                
                # --- Lógica de cálculo y botones (Corregido el error de StreamlitAPIException) ---

                if "Intervalo de Confianza" in opcion_dos:
                    confianza_icd = st.slider("Nivel de Confianza (%)", min_value=80, max_value=99, value=95, key='conf_icd') / 100.0
                    
                    if opcion_dos == "Diferencia de Medias (Intervalo de Confianza)":
                        # Se leen las variables solo si están definidas en el bloque superior (Media)
                        if st.button("Calcular IC (Medias)", key='btn_ic_medias'):
                            st.session_state.update({
                                'tipo_calculo_dos': 'IC_Medias', 'confianza_icd': confianza_icd,
                                'media1': media1, 'desv1': desv1, 'n1': n1,
                                'media2': media2, 'desv2': desv2, 'n2': n2
                            })
                    
                    elif opcion_dos == "Diferencia de Proporciones (Intervalo de Confianza)":
                        if st.button("Calcular IC (Proporciones)", key='btn_ic_props'):
                            # Se leen las variables solo si están definidas en el bloque superior (Proporción)
                            st.session_state.update({
                                'tipo_calculo_dos': 'IC_Proporciones', 'confianza_icd': confianza_icd,
                                'x1': x1, 'n1': n1, 'x2': x2, 'n2': n2
                            })

                elif "Prueba de Hipótesis" in opcion_dos:
                    alfa_ph = st.slider("Nivel de Significación (α) (%)", min_value=1, max_value=10, value=5, key='alfa_ph_dos') / 100.0
                    tipo_ph = st.radio("Tipo de Prueba H₁:", ["≠ (Dos colas)", "< (Cola izquierda)", "> (Cola derecha)"], key='tipo_ph_dos')

                    if opcion_dos == "Prueba de Hipótesis para Medias":
                        if st.button("Realizar PH (Medias)", key='btn_ph_medias'):
                            # Se leen las variables solo si están definidas en el bloque superior (Media)
                            st.session_state.update({
                                'tipo_calculo_dos': 'PH_Medias', 'alfa_ph': alfa_ph, 'tipo_ph': tipo_ph,
                                'media1': media1, 'desv1': desv1, 'n1': n1,
                                'media2': media2, 'desv2': desv2, 'n2': n2
                            })
                    
                    elif opcion_dos == "Prueba de Hipótesis para Proporciones":
                        if st.button("Realizar PH (Proporciones)", key='btn_ph_props'):
                            # Se leen las variables solo si están definidas en el bloque superior (Proporción)
                            st.session_state.update({
                                'tipo_calculo_dos': 'PH_Proporciones', 'alfa_ph': alfa_ph, 'tipo_ph': tipo_ph,
                                'x1': x1, 'n1': n1, 'x2': x2, 'n2': n2
                            })


            with result_dos:
                st.subheader("Resultados de Dos Poblaciones (6ta Pestaña)")
                
                # Verificación robusta del estado
                tipo_calc_dos = st.session_state.get('tipo_calculo_dos')
                
                # --- Resultados: IC Diferencia de Medias ---
                if tipo_calc_dos == 'IC_Medias':
                    conf = st.session_state['confianza_icd']
                    media1, desv1, n1 = st.session_state['media1'], st.session_state['desv1'], st.session_state['n1']
                    media2, desv2, n2 = st.session_state['media2'], st.session_state['desv2'], st.session_state['n2']

                    Z = get_z_score(conf)
                    diferencia_muestral = media1 - media2
                    ee_diferencia = np.sqrt((desv1**2 / n1) + (desv2**2 / n2))
                    margen_error = Z * ee_diferencia

                    lim_inf = diferencia_muestral - margen_error
                    lim_sup = diferencia_muestral + margen_error

                    st.success(f"El IC ({conf*100}%) para $\\mu_1 - \\mu_2$ es: **[{lim_inf:,.4f}, {lim_sup:,.4f}]**")
                    st.write(f"**Diferencia de Medias (x̄₁ - x̄₂):** {diferencia_muestral:,.4f}")
                    st.write(f"**Error Estándar de la Diferencia (EE):** {ee_diferencia:,.4f}")
                    st.write(f"**Margen de Error (E):** {margen_error:,.4f}")
                    st.write(f"*(Si el intervalo contiene al 0, no hay evidencia de diferencia.)*")
                
                # --- Resultados: IC Diferencia de Proporciones ---
                elif tipo_calc_dos == 'IC_Proporciones':
                    conf = st.session_state['confianza_icd']
                    x1, n1, x2, n2 = st.session_state['x1'], st.session_state['n1'], st.session_state['x2'], st.session_state['n2']
                    
                    p1_hat, p2_hat = x1 / n1, x2 / n2
                    Z = get_z_score(conf)
                    diferencia_muestral = p1_hat - p2_hat
                    
                    ee_diferencia = np.sqrt((p1_hat * (1 - p1_hat) / n1) + (p2_hat * (1 - p2_hat) / n2))
                    margen_error = Z * ee_diferencia
                    
                    lim_inf = diferencia_muestral - margen_error
                    lim_sup = diferencia_muestral + margen_error
                    
                    st.success(f"El IC ({conf*100}%) para $p_1 - p_2$ es: **[{lim_inf:,.4f}, {lim_sup:,.4f}]**")
                    st.write(f"**Diferencia de Proporciones (p̂₁ - p̂₂):** {diferencia_muestral:,.4f}")
                    st.write(f"**Error Estándar de la Diferencia (EE):** {ee_diferencia:,.4f}")
                
                # --- Resultados: Prueba de Hipótesis para Medias ---
                elif tipo_calc_dos == 'PH_Medias':
                    alfa_ph = st.session_state['alfa_ph']
                    tipo_ph = st.session_state['tipo_ph']
                    media1, desv1, n1 = st.session_state['media1'], st.session_state['desv1'], st.session_state['n1']
                    media2, desv2, n2 = st.session_state['media2'], st.session_state['desv2'], st.session_state['n2']

                    # Cálculo Z (asumiendo n grande - TLC)
                    ee_diferencia = np.sqrt((desv1**2 / n1) + (desv2**2 / n2))
                    z_calculado = (media1 - media2) / ee_diferencia
                    
                    # Cálculo del Valor P
                    if "Dos colas" in tipo_ph:
                        valor_p = 2 * (1 - stats.norm.cdf(abs(z_calculado)))
                    elif "Cola izquierda" in tipo_ph:
                        valor_p = stats.norm.cdf(z_calculado)
                    else: # Cola derecha
                        valor_p = 1 - stats.norm.cdf(z_calculado)
                        
                    st.info(f"Hipótesis Nula ($H_0$): $\\mu_1 - \\mu_2 = 0$. Hipótesis Alternativa ($H_1$): $\\mu_1 - \\mu_2 {tipo_ph[0]}$ $0$")
                    st.write(f"**Estadístico de Prueba (Z):** {z_calculado:,.4f}")
                    st.write(f"**Valor P:** {valor_p:,.4f}")
                    st.write(f"**Nivel de Significación (α):** {alfa_ph}")

                    st.markdown("---")
                    
                    if valor_p < alfa_ph:
                        st.error("Decisión: **Rechazar $H_0$**.")
                        st.write("Existe evidencia suficiente ($P < α$) para concluir que hay una diferencia significativa.")
                    else:
                        st.success("Decisión: **No Rechazar $H_0$**.")
                        st.write("No existe evidencia suficiente ($P \ge α$) para concluir que hay una diferencia significativa.")
                        
                    st.markdown("---")
                    
                    # Extra Opcional: Teorema del Límite Central (TLC)
                    st.subheader("Extra Opcional: Teorema del Límite Central (TLC)")
                    st.markdown("El cálculo de $Z$ se basa en el **TLC**, asumiendo una distribución normal para las medias muestrales grandes, permitiendo usar la curva normal estándar para el cálculo del valor P.")
                    
                # --- Resultados: Prueba de Hipótesis para Proporciones ---
                elif tipo_calc_dos == 'PH_Proporciones':
                    alfa_ph = st.session_state['alfa_ph']
                    tipo_ph = st.session_state['tipo_ph']
                    x1, n1, x2, n2 = st.session_state['x1'], st.session_state['n1'], st.session_state['x2'], st.session_state['n2']
                    
                    p1_hat, p2_hat = x1 / n1, x2 / n2
                    p_pooled = (x1 + x2) / (n1 + n2) # Proporción combinada para PH
                    
                    # Cálculo del Estadístico Z
                    ee_ph = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                    z_calculado = (p1_hat - p2_hat) / ee_ph
                    
                    # Cálculo del Valor P
                    if "Dos colas" in tipo_ph:
                        valor_p = 2 * (1 - stats.norm.cdf(abs(z_calculado)))
                    elif "Cola izquierda" in tipo_ph:
                        valor_p = stats.norm.cdf(z_calculado)
                    else: # Cola derecha
                        valor_p = 1 - stats.norm.cdf(z_calculado)
                        
                    st.info(f"Hipótesis Nula ($H_0$): $p_1 - p_2 = 0$. Hipótesis Alternativa ($H_1$): $p_1 - p_2 {tipo_ph[0]}$ $0$")
                    st.write(f"**Proporción combinada (p̄):** {p_pooled:,.4f}")
                    st.write(f"**Estadístico de Prueba (Z):** {z_calculado:,.4f}")
                    st.write(f"**Valor P:** {valor_p:,.4f}")
                    st.write(f"**Nivel de Significación (α):** {alfa_ph}")

                    st.markdown("---")
                    
                    if valor_p < alfa_ph:
                        st.error("Decisión: **Rechazar $H_0$**.")
                        st.write("Existe evidencia suficiente ($P < α$) para concluir que la diferencia de proporciones es significativa.")
                    else:
                        st.success("Decisión: **No Rechazar $H_0$**.")
                        st.write("No existe evidencia suficiente ($P \ge α$) para concluir que hay una diferencia significativa.")

                elif tipo_calc_dos is None:
                    st.info("El resultado del cálculo de 'Dos Poblaciones' aparecerá aquí.")


# --- EJECUTAR LA APP ---
if __name__ == "__main__":
    main()
                    
