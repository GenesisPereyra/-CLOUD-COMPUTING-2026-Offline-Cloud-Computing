# Federated Learning with MNIST

## Equipo 2

Equipo: Chipocles

Integrantes:
- Jose de Jesus Ramirez Mendieta | A00835680
- Maritza Barrios Macías | A00836821
- Máximo Caballero Vargas | A01571607
- Genesis Pereyra Camacho | A01734276
- Luis Roberto Garza Sánchez | A00836982

## Descripción

El modelo utilizado fue una red neuronal densa o MLP, distinta al modelo CNN visto en clase. La imagen de entrada de 28x28 píxeles se aplana mediante una capa Flatten y después pasa por dos capas densas de 128 y 64 neuronas con activación ReLU. Finalmente, una capa Softmax de 10 neuronas clasifica los dígitos del 0 al 9.

## Estructura del proyecto

```text
TheModel.py
local_training.ipynb
global_aggregation.py
requirements.txt
README.md
local_models/
global_models/
reports/
figures/


FedAvg:
FedAvg calcula el promedio directo de los pesos de todos los modelos locales. Es el método base de aprendizaje federado. Funciona bien cuando todos los clientes tienen una cantidad similar de datos y distribuciones parecidas.

Weighted FedAvg:
Weighted FedAvg es una variante de FedAvg en donde los modelos locales no se promedian todos con el mismo peso. En FedAvg tradicional, si hay 5 clientes, cada uno podría aportar 1/5 al modelo global. En cambio, en Weighted FedAvg se asigna un peso diferente a cada cliente, normalmente según la cantidad de datos, desempeño local o alguna medida de confiabilidad. Esto se relaciona con el artículo de Adaptive Weighted Aggregation, donde se plantea que una estrategia adecuada de agregación en el servidor puede reducir el efecto de la heterogeneidad entre clientes y mejorar la convergencia del aprendizaje federado. El paper también menciona que FedAWARE ajusta los pesos de agregación usando actualizaciones locales y busca mejorar la generalización bajo clientes heterogéneos.

Trimmed Mean:
Trimmed Mean es una técnica robusta de agregación. En lugar de promediar directamente todos los valores, primero elimina una proporción de valores extremos y después promedia lo restante. El paper de Depth based trimmed means explica que las medias recortadas ayudan a reducir la influencia de valores extremos u outliers, lo cual puede hacer que la estimación sea más robusta cuando hay datos contaminados, errores o valores anómalos.

En aprendizaje federado, esta idea se puede adaptar a los pesos de los modelos: para cada parámetro de cada capa, se ordenan los valores provenientes de los clientes, se eliminan los valores más bajos y más altos, y luego se promedia lo restante. Esto puede ayudar si un cliente entrenó mal, tuvo datos raros o generó pesos muy diferentes al resto.