import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats # Necesario para Z, T-student, Intervalos de Confianza

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Calculadora Estadística",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCIÓN PRINCIPAL DE LA APP ---
def main():
    st.title("📊 Calculadora Estadística Avanzada")
    st.markdown("---")

    # --- NAVEGACIÓN PRINCIPAL (Sidebar) ---
    # Usamos la barra lateral para las grandes secciones (Tendencia Central, Inferencia)
    st.sidebar.title("Menú Principal")
    seccion = st.sidebar.radio(
        "Selecciona una Sección:",
        ["Medidas de Tendencia Central", "Inferencia Estadística"]
    )

    # --- SECCIÓN 1: Medidas de Tendencia Central ---
    if seccion == "Medidas de Tendencia Central":
        st.header("1️⃣ Medidas de Tendencia Central")

        # Pestañas para Input y Resultado
        tab_input, tab_resultados = st.tabs(["Datos de Entrada", "Resultados"])

        with tab_input:
            st.subheader("Ingreso de Datos")
            datos_raw = st.text_area(
                "Ingresa tus datos separados por comas, espacios o saltos de línea (ej: 1, 5, 10, 15, 20)",
                height=150
            )

            # Procesa los datos
            if datos_raw:
                try:
                    # Limpiar y convertir los datos a números
                    datos_list = [float(x.strip()) for x in datos_raw.replace(',', ' ').split() if x.strip()]
                    datos = np.array(datos_list)
                    calcular_tendencia = st.button("Calcular Medidas")
                except ValueError:
                    st.error("Error: Asegúrate de que los datos sean números válidos.")
                    datos = None
                    calcular_tendencia = False
            else:
                datos = None
                calcular_tendencia = False

        with tab_resultados:
            st.subheader("Resultados del Análisis Descriptivo")
            if calcular_tendencia and datos is not None:
                if len(datos) > 0:
                    # Cálculo de Medidas de Tendencia Central
                    media = np.mean(datos)
                    mediana = np.median(datos)
                    moda_res = stats.mode(datos, keepdims=False)
                    moda = moda_res.mode if moda_res.count > 0 else "No hay moda clara"

                    st.metric("Media (Promedio)", f"{media:,.4f}")
                    st.metric("Mediana", f"{mediana:,.4f}")
                    st.metric("Moda", f"{moda}")

                    st.markdown("---")

                    # Extra Opcional: Histograma (Se adapta muy bien a la Tendencia Central)
                    st.subheader("Extra Opcional: Distribución de Frecuencias (Histograma)")
                    fig, ax = plt.subplots()
                    ax.hist(datos, bins=10, edgecolor='black', alpha=0.7)
                    ax.set_title('Histograma de los Datos')
                    ax.set_xlabel('Valor')
                    ax.set_ylabel('Frecuencia')
                    st.pyplot(fig) # Muestra el gráfico en Streamlit
                else:
                    st.warning("Por favor, ingresa datos válidos para calcular.")

    # --- SECCIÓN 2: Inferencia Estadística ---
    elif seccion == "Inferencia Estadística":
        st.header("2️⃣ Inferencia Estadística")

        # Pestañas para Una Población y Dos Poblaciones
        tabs_poblacion = st.tabs(["3️⃣ Una Población", "5️⃣ Dos Poblaciones"])

        # --- SUBSECCIÓN: UNA POBLACIÓN ---
        with tabs_poblacion[0]:
            st.subheader("Cálculos para Una Única Población")
            
            # Sub-pestañas para cada cálculo y sus resultados
            calc_una, result_una = st.tabs(["Calculadora", "Resultados (4ta Pestaña)"])

            with calc_una:
                opcion_una = st.selectbox(
                    "Selecciona el Cálculo:",
                    [
                        "Error Estándar de la Media",
                        "Intervalo de Confianza de la Media (μ)",
                        "Intervalo de Confianza de una Proporción (p)",
                        "Cálculo de Z y T-student",
                        "Tamaño de Muestra por Media",
                        "Tamaño de Muestra por Proporción"
                    ]
                )
                
                # --- Lógica de inputs para la opción_una ---
                st.info(f"Inputs para: **{opcion_una}**")
                
                # Ejemplo de input para Intervalo de Confianza de la Media
                if "Intervalo de Confianza de la Media" in opcion_una:
                    media_m = st.number_input("Media muestral (x̄)", value=50.0)
                    desv_std = st.number_input("Desviación estándar de la muestra (s)", value=10.0)
                    n = st.number_input("Tamaño de la muestra (n)", min_value=1, value=30)
                    confianza = st.slider("Nivel de Confianza (%)", min_value=80, max_value=99, value=95) / 100.0
                    
                    if st.button("Calcular Intervalo (Media)"):
                        # Pasar resultados a la pestaña de Resultados
                        st.session_state['resultado_una'] = f"Intervalo de Confianza de la Media calculado."
                        st.session_state['valor_confianza'] = confianza
                        st.session_state['n_media'] = n
                        st.session_state['media_m'] = media_m
                        st.session_state['desv_std'] = desv_std
                        st.session_state['tipo_calculo_una'] = 'IC_Media'

            with result_una:
                if 'tipo_calculo_una' in st.session_state and st.session_state['tipo_calculo_una'] == 'IC_Media':
                    st.subheader("Intervalo de Confianza de la Media (μ)")
                    
                    media_m = st.session_state['media_m']
                    desv_std = st.session_state['desv_std']
                    n = st.session_state['n_media']
                    confianza = st.session_state['valor_confianza']
                    alfa = 1 - confianza
                    
                    # Calcular T-student (asumiendo n < 30 o sigma desconocida)
                    grados_libertad = n - 1
                    t_score = stats.t.ppf(1 - alfa/2, grados_libertad)
                    
                    error_estandar_ic = desv_std / np.sqrt(n)
                    margen_error = t_score * error_estandar_ic
                    
                    lim_inf = media_m - margen_error
                    lim_sup = media_m + margen_error
                    
                    st.write(f"**Nivel de Confianza:** {confianza*100}%")
                    st.write(f"**Error Estándar (EE):** {error_estandar_ic:,.4f}")
                    st.write(f"**Estadístico T-Student:** {t_score:,.4f} (gl={grados_libertad})")
                    st.write(f"**Margen de Error:** {margen_error:,.4f}")
                    st.success(f"El Intervalo de Confianza es: **[{lim_inf:,.4f}, {lim_sup:,.4f}]**")

                    st.markdown("---")
                    
                    # Extra Opcional: Distribución muestral (Comportamiento del EE)
                    st.subheader("Extra Opcional: Ilustración del Error Estándar (EE)")
                    st.markdown("El **Error Estándar** ($EE = s/\sqrt{n}$) mide la variabilidad de la media muestral respecto a la media poblacional. A medida que $n$ aumenta, el $EE$ se reduce y el intervalo se estrecha.")
                    #  # Sugerir una imagen del TLC mostrando la reducción de la dispersión
                    
                else:
                    st.info("El resultado del cálculo de 'Una Población' aparecerá aquí.")

    # --- SUBSECCIÓN: DOS POBLACIONES ---
        with tabs_poblacion[1]:
            st.subheader("Cálculos para Dos Poblaciones (Comparación)")

            # Sub-pestañas para cada cálculo y sus resultados
            calc_dos, result_dos = st.tabs(["Calculadora", "Resultados (6ta Pestaña)"])

            with calc_dos:
                opcion_dos = st.selectbox(
                    "Selecciona el Cálculo:",
                    [
                        "Diferencia de Medias",
                        "Diferencia de Proporciones",
                        "Prueba de Hipótesis para Medias",
                        "Prueba de Hipótesis para Proporciones"
                    ]
                )
                
                # --- Lógica de inputs para la opción_dos ---
                st.info(f"Inputs para: **{opcion_dos}**")
                
                # Ejemplo de input para Prueba de Hipótesis para Medias
                if "Prueba de Hipótesis para Medias" in opcion_dos:
                    st.markdown("#### Población 1")
                    media1 = st.number_input("Media muestral 1 (x̄₁)", value=60.0, key='m1')
                    desv1 = st.number_input("Desviación estándar 1 (s₁)", value=8.0, key='d1')
                    n1 = st.number_input("Tamaño de la muestra 1 (n₁)", min_value=1, value=40, key='n1')

                    st.markdown("#### Población 2")
                    media2 = st.number_input("Media muestral 2 (x̄₂)", value=55.0, key='m2')
                    desv2 = st.number_input("Desviación estándar 2 (s₂)", value=7.0, key='d2')
                    n2 = st.number_input("Tamaño de la muestra 2 (n₂)", min_value=1, value=35, key='n2')

                    alfa_ph = st.slider("Nivel de Significación (α)", min_value=1, max_value=10, value=5) / 100.0
                    
                    if st.button("Realizar Prueba de Hipótesis (Medias)"):
                        st.session_state['tipo_calculo_dos'] = 'PH_Medias'
                        st.session_state['media1'] = media1
                        st.session_state['desv1'] = desv1
                        st.session_state['n1'] = n1
                        st.session_state['media2'] = media2
                        st.session_state['desv2'] = desv2
                        st.session_state['n2'] = n2
                        st.session_state['alfa_ph'] = alfa_ph

            with result_dos:
                if 'tipo_calculo_dos' in st.session_state and st.session_state['tipo_calculo_dos'] == 'PH_Medias':
                    st.subheader("Prueba de Hipótesis para la Diferencia de Medias")
                    
                    # Recuperar valores
                    media1, desv1, n1 = st.session_state['media1'], st.session_state['desv1'], st.session_state['n1']
                    media2, desv2, n2 = st.session_state['media2'], st.session_state['desv2'], st.session_state['n2']
                    alfa_ph = st.session_state['alfa_ph']
                    
                    # Cálculo de Estadístico Z (asumiendo n1 y n2 > 30, o varianzas conocidas)
                    # Usaremos Z para simplicidad, pero T-student es más riguroso si n < 30 y varianza desconocida.
                    
                    # Cálculo de la varianza combinada (si se asume varianzas iguales y T-test, pero simplificamos con Z)
                    # Usaremos Z, asumiendo muestras grandes (CLT) o varianzas poblacionales conocidas.
                    ee_diferencia = np.sqrt((desv1**2 / n1) + (desv2**2 / n2))
                    z_calculado = (media1 - media2) / ee_diferencia
                    
                    # Valor P (prueba de dos colas por defecto, es lo más común)
                    valor_p = 2 * (1 - stats.norm.cdf(abs(z_calculado)))
                    
                    st.write(f"**Diferencia de Medias (x̄₁ - x̄₂):** {media1 - media2:,.4f}")
                    st.write(f"**Error Estándar de la Diferencia (EE):** {ee_diferencia:,.4f}")
                    st.write(f"**Estadístico de Prueba (Z):** {z_calculado:,.4f}")
                    st.write(f"**Valor P:** {valor_p:,.4f}")
                    st.write(f"**Nivel de Significación (α):** {alfa_ph}")

                    st.markdown("---")
                    
                    if valor_p < alfa_ph:
                        st.error("Decisión: **Rechazar $H_0$**.")
                        st.write("Existe evidencia suficiente ($P < α$) para concluir que hay una diferencia significativa entre las medias de las dos poblaciones.")
                    else:
                        st.success("Decisión: **No Rechazar $H_0$**.")
                        st.write("No existe evidencia suficiente ($P \ge α$) para concluir que hay una diferencia significativa entre las medias de las dos poblaciones.")
                        
                    st.markdown("---")

                    # Extra Opcional: Teorema del Límite Central (TLC) - Se adapta al cálculo de Z/T
                    st.subheader("Extra Opcional: Teorema del Límite Central (TLC)")
                    st.markdown("El cálculo de la $Z$ se basa en el **TLC**, que establece que la distribución de las diferencias de medias muestrales tiende a una distribución normal si los tamaños de muestra ($n_1$ y $n_2$) son suficientemente grandes (generalmente $n \ge 30$).")
                    st.markdown("Distribución normal para Z: ") # Diagrama de la curva normal de Z con regiones de rechazo
                    
                else:
                    st.info("El resultado del cálculo de 'Dos Poblaciones' aparecerá aquí.")

    # Importar pyplot de matplotlib para el Histograma
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("Error: La librería 'matplotlib' no está instalada. Ejecuta 'pip install matplotlib'.")
    plt = None


    # --- EJECUTAR LA APP ---
if __name__ == "__main__":
    if 'resultado_una' not in st.session_state:
        st.session_state['resultado_una'] = None
    if 'resultado_dos' not in st.session_state:
        st.session_state['resultado_dos'] = None
    
    main()

