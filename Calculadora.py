import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Calculadora Estadística Pro", layout="wide")

# --- LÓGICA MATEMÁTICA ---
def get_z_critico(confianza):
    return stats.norm.ppf(1 - (1 - confianza) / 2)

def get_t_critico(confianza, gl):
    return stats.t.ppf(1 - (1 - confianza) / 2, df=gl)

# --- VISTA: TENDENCIA CENTRAL ---
def seccion_tendencia_central():
    st.header("🎯 Medidas de Tendencia Central")
    col1, col2 = st.columns([1, 1])
    with col1:
        datos_input = st.text_area("Ingresa datos (separados por espacio o coma):", height=150)
        btn_calc = st.button("Analizar Datos")
    
    if btn_calc and datos_input:
        try:
            arr = np.array([float(x.strip()) for x in datos_input.replace(',', ' ').split() if x.strip()])
            with col2:
                st.subheader("📊 Resultados")
                res = {
                    "Media": np.mean(arr),
                    "Mediana": np.median(arr),
                    "Moda": stats.mode(arr, keepdims=False).mode,
                    "Desv. Estándar (s)": np.std(arr, ddof=1),
                    "Recuento (n)": len(arr)
                }
                st.table(pd.DataFrame([res]).T.rename(columns={0: "Valor"}))
                fig, ax = plt.subplots()
                ax.hist(arr, bins='auto', color='#3498db', edgecolor='white')
                st.pyplot(fig)
        except Exception as e:
            st.error("Error al procesar los datos. Asegúrate de ingresar solo números.")

# --- VISTA: UNA POBLACIÓN ---
def seccion_inferencia_1_pop():
    st.header("👤 Inferencia: Una Población")
    opcion = st.selectbox("Selecciona el cálculo:", [
        "Error estándar de la media",
        "Intervalo de confianza de la media",
        "Intervalo de confianza de una proporción",
        "Cálculo de Z y T-student",
        "Tamaño de muestra por media",
        "Tamaño de muestra por proporción"
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        if opcion == "Error estándar de la media":
            s = st.number_input("Desviación estándar (s)", value=1.0)
            n = st.number_input("n", value=30, min_value=1)
            if st.button("Calcular"):
                ee = s / np.sqrt(n)
                col2.metric("Error Estándar (EE)", f"{ee:.4f}")

        elif opcion == "Intervalo de confianza de la media":
            x_bar = st.number_input("Media muestral (x̄)", value=50.0)
            s = st.number_input("Desviación estándar (s)", value=5.0)
            n = st.number_input("n", value=30, min_value=2)
            conf = st.slider("Confianza", 0.80, 0.99, 0.95)
            if st.button("Calcular IC"):
                t_crit = get_t_critico(conf, n-1)
                err = t_crit * (s / np.sqrt(n))
                col2.success(f"IC: [{x_bar-err:.4f}, {x_bar+err:.4f}]")

        elif opcion == "Intervalo de confianza de una proporción":
            x = st.number_input("Éxitos (x)", value=15)
            n = st.number_input("Total (n)", value=50)
            conf = st.slider("Confianza", 0.80, 0.99, 0.95)
            if st.button("Calcular IC Prop"):
                p_hat = x / n
                z_crit = get_z_critico(conf)
                err = z_crit * np.sqrt((p_hat * (1-p_hat))/n)
                col2.success(f"IC: [{p_hat-err:.4f}, {p_hat+err:.4f}]")

        elif opcion == "Cálculo de Z y T-student":
            x_bar = st.number_input("Media muestral (x̄)", value=10.0)
            mu = st.number_input("Media hipotética (μ)", value=8.0)
            s = st.number_input("Desviación estándar (s)", value=2.0)
            n = st.number_input("n", value=25)
            if st.button("Calcular Estadísticos"):
                z_t = (x_bar - mu) / (s / np.sqrt(n))
                col2.metric("Estadístico Calculado", f"{z_t:.4f}")

        elif opcion == "Tamaño de muestra por media":
            conf = st.slider("Confianza", 0.80, 0.99, 0.95)
            s = st.number_input("Desviación estándar estimada", value=1.0)
            e = st.number_input("Margen de error deseado (E)", value=0.1)
            if st.button("Calcular n"):
                z = get_z_critico(conf)
                n = (z * s / e)**2
                col2.metric("n requerido", int(np.ceil(n)))

        elif opcion == "Tamaño de muestra por proporción":
            conf = st.slider("Confianza", 0.80, 0.99, 0.95)
            p = st.number_input("Proporción estimada (p)", 0.0, 1.0, 0.5)
            e = st.number_input("Margen de error (E)", value=0.05)
            if st.button("Calcular n"):
                z = get_z_critico(conf)
                n = (z**2 * p * (1-p)) / (e**2)
                col2.metric("n requerido", int(np.ceil(n)))

# --- VISTA: DOS POBLACIONES ---
def seccion_inferencia_2_pops():
    st.header("👥 Inferencia: Dos Poblaciones")
    opcion = st.selectbox("Selecciona el cálculo:", [
        "Diferencia de medias (IC)",
        "Diferencia de proporciones (IC)",
        "Prueba de hipótesis para medias",
        "Prueba de hipótesis para proporciones"
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Muestra 1")
        n1 = st.number_input("n1", value=30, key="n1")
        if "proporciones" in opcion.lower():
            x1 = st.number_input("Éxitos x1", value=15.0, key="x1")
        else:
            m1 = st.number_input("Media x̄1", value=10.0, key="m1")
            s1 = st.number_input("Desv. Est. s1", value=2.0, key="s1")

    with col2:
        st.subheader("Muestra 2")
        n2 = st.number_input("n2", value=30, key="n2")
        if "proporciones" in opcion.lower():
            x2 = st.number_input("Éxitos x2", value=20.0, key="x2")
        else:
            m2 = st.number_input("Media x̄2", value=12.0, key="m2")
            s2 = st.number_input("Desv. Est. s2", value=2.5, key="s2")

    st.divider()
    
    if "IC" in opcion:
        conf = st.slider("Confianza", 0.80, 0.99, 0.95)
        if st.button("Calcular Intervalo"):
            z = get_z_critico(conf)
            if "medias" in opcion.lower():
                diff = m1 - m2
                ee = np.sqrt((s1**2/n1) + (s2**2/n2))
            else:
                p1, p2 = x1/n1, x2/n2
                diff = p1 - p2
                ee = np.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
            err = z * ee
            st.success(f"Intervalo de confianza para la diferencia: [{diff-err:.4f}, {diff+err:.4f}]")

    elif "Prueba" in opcion:
        h1 = st.selectbox("H1:", ["≠ (Diferente)", "> (Mayor)", "< (Menor)"])
        alfa = st.number_input("Alfa (α)", value=0.05)
        if st.button("Ejecutar Prueba"):
            if "medias" in opcion.lower():
                ee = np.sqrt((s1**2/n1) + (s2**2/n2))
                z_calc = (m1 - m2) / ee
            else:
                p1, p2, pc = x1/n1, x2/n2, (x1+x2)/(n1+n2)
                ee = np.sqrt(pc * (1-pc) * (1/n1 + 1/n2))
                z_calc = (p1 - p2) / ee
            
            p_val = stats.norm.sf(abs(z_calc)) * (2 if "≠" in h1 else 1)
            st.metric("Estadístico Z", f"{z_calc:.4f}")
            st.metric("Valor p", f"{p_val:.4f}")
            if p_val < alfa: st.error("Rechazamos H0: Hay diferencia significativa.")
            else: st.success("No rechazamos H0: No hay diferencia significativa.")

# --- MAIN ---
def main():
    st.sidebar.title("Calculadora Estadística")
    menu = st.sidebar.radio("Menú:", ["Inicio", "Tendencia Central", "Una Población", "Dos Poblaciones"])
    
    if menu == "Inicio":
        st.title("🧮 Bienvenido")
        st.write("Esta calculadora permite realizar análisis descriptivos e inferenciales de forma eficiente.")
        st.info("Selecciona una sección en el menú lateral para comenzar.")
    elif menu == "Tendencia Central":
        seccion_tendencia_central()
    elif menu == "Una Población":
        seccion_inferencia_1_pop()
    elif menu == "Dos Poblaciones":
        seccion_inferencia_2_pops()

if __name__ == "__main__":
    main()
    
