# Caracterización cinemática de la función motriz manual mediante visión artificial

**Código y datos del trabajo de grado en Ingeniería Mecatrónica, Universidad de San Buenaventura (Bogotá, 2023).**

*English summary: Python pipeline and dataset for measuring hand joint angles and kinematics from ordinary video, using MediaPipe hand tracking. Data from 31 healthy volunteers (186 videos, two camera views), with statistical analysis in SPSS and 3D simulation in Blender.*

**Autores:** Juan Felipe Martín Martínez, Laura Alejandra Gómez Rodríguez, Juan David Cruz Contreras
**Asesor:** Ing. Erick Libardo Piñeros Valbuena, MSc
**Mi contribución:** Desarrollo del código, ejecución de las pruebas junto al equipo y análisis de datos en Python. 

---

## Problema

La evaluación de la motricidad fina suele depender de laboratorios con marcadores infrarrojos (costosos e incómodos para el paciente) o de observación clínica subjetiva. Este proyecto explora una alternativa: medir el movimiento de la mano con cámaras comunes y visión artificial.

## Qué se hizo

- Diseño de un protocolo biomecánico con tres pruebas: rasgado de papel, esfera de plastilina y compresión de plastilina.
- Grabación de **186 videos** de **31 voluntarios sanos** (18 a 30 años) desde dos vistas: frontal a 37° y superior a 90°, a 30 FPS.
- Extracción de puntos de la mano con **MediaPipe** y cálculo de ángulos articulares, velocidad y aceleración con **Python**.
- Depuración de valores atípicos: se conservó el 82 % de los datos en la vista de 37° y el 92 % en la vista superior.
- Consistencia interna (alfa de Cronbach): **0,851** (vista 37°) y **0,815** (vista superior).
- Comparaciones por grupo de edad y por mano dominante (SPSS) y simulación 3D de las posiciones de la mano en Blender.

## Estructura del repositorio

| Carpeta / archivo | Contenido |
|---|---|
| `FRONTAL_37°/` | Código de análisis de la vista frontal a 37° y bases de datos (.xlsx, .spv) |
| `HORIZONTAL_90°/` | Código de análisis de la vista superior a 90° y bases de datos (.xlsx, .spv) |
| `Comparaciones edades/` | Comparación de rangos articulares entre grupos de edad |
| `Diestros vs Zurdos/` | Comparación entre mano dominante y no dominante zurdos y diestros |
| `Velocidad y aceleración/` + `Codigo velocidad aceleracion` | Análisios de parámetros cinemáticos de los dedos |
| `Trayectoria prueba 3/` + `Codigo trayectoria` | Trayectorias en la prueba de compresión |
| `Codigo espacio de trabajo` | Cálculo del espacio de trabajo de la mano |
| `IMAGENES_SPSS/` | Salidas gráficas del análisis estadístico |
| `Diagrama y QR/` | Diagramas del proceso |

**Entradas:** videos de las pruebas (no incluidos en este repositorio). **Salidas:** tablas de ángulos por articulación (.xlsx) y gráficas.

## Limitaciones

- Muestra de personas sanas de 18 a 30 años; los resultados no se generalizan a población con patologías o adultos mayores.
- Cámaras de celular a 30 FPS: el seguimiento de MediaPipe mejora con movimientos lentos y la captura de movimientos rápidos se degrada.
- No se validó contra un sistema de referencia (p. ej. captura con marcadores).
- Fuera de alcance: abducción y aducción de las falanges.

## Trabajo futuro

Validación contra un estándar de referencia, mayor frecuencia de captura, inclusión de pacientes y aplicación a evaluación remota (telemedicina).

## Datos y ética

Los participantes firmaron consentimiento informado y fueron identificados según su edad.

## Cómo citar

Cruz, J. D., Gómez, L. A., & Martín, J. F. (2023). *Caracterización cinemática de la función motriz manual a través de técnicas de visión de máquina*. Trabajo de grado, Ingeniería Mecatrónica, Universidad de San Buenaventura, Bogotá.
