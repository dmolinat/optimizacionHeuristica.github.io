<div align="center">
  
# Trabajo 1: Optimizacion Heurística.

## Redes neuronales y algorítmos bio-inspirados.
## 2024-2   

## Denilson Andrés Molina Truyot
## Universidad Nacional de Colombia
## Sede Medellín
</div>

## 1. Optimización de funciones.

Las funciones elegidas para realizar los siguientes items son: **Función Goldstein-Price** y **Función de Griewank**. Ambas funciones se presentarán para recibir 2 y 3 componentes. Además la función Goldstein-Price no posee extensión a N dimensiones, por lo que se hizo una aproximación teniendo presente el comportamiento en la definición original de esta.

Las implementaciones a usar, se realizarán en Python, y los códigos claves para ello se irán mostrando a medida que se desarrolla el contenido. Además, para optimizar dichas funciones se utilizarán los siguientes métodos:

### 1.1. Optimización numérica por descenso del gradiente.

El descenso de gradiente consta de la siguiente idea: Para encontrar el mínimo de una función, debemos movernos en la dirección opuesta al gradiente de la función en el punto actual (Goodfellow, Bengio, & Courville, 2016, p. 96). El gradiente de una función es un **vector** que contiene las derivadas parciales de la función con respecto a todas sus variables. Indica la dirección de mayor aumento de la función, por lo que moverse en la dirección opuesta al gradiente conduce a la mayor diminusión de esta. Consta de los siguientes pasos.

#### 1.1.1. Inicialización. 

Comienza con un punto inicial aleatorio o específico en el espacio de parámetros. Esto se implementa dependiendo el dominio de interés que se desea estudiar. Para lograr ello se realizará la siguiente función que busca generar un vector, en este caso array, dado las dimensiones y el dominio de la cuadricula que se desee estudiar. 

```python
def gen_rand_X(a,b,N=1,dim=2):
  return np.random.uniform(a, b, size=(N, dim))
```

Siendo *N* el numero de vectores a generar y *dim*, el número de componentes de cada vector en un dominio [a,b].

#### 1.1.2. Calcular gradiente.

Dicho vector, contiene todas las derivadas parciales de cada variable. En este caso se implementa basado en un código reciclado del cuaderno (Zapata & Ospina, 2024).

```python

def partial_dev(x0,f,i,h=0.0001):
  # i: es la i-ésima componente del gradiente y comienza en 0
  e = x0*0.0
  e[i] = h
  y = (f(x0+e) - f(x0-e))/(2*h)
  return(y)

def grad_num(x0,f,h=0.00001):
  y = [partial_dev(x0,f,h,i) for i in range(0,len(x0))]
  y = np.array(y)
  return(y)
```

#### 1.1.3. Definición de iteraciones.

Los vectores que se acercan a una solución óptima se actualizan moviéndose en la dirección opuesta al gradiente. La magnitud del movimiento está determinada por un factor llamado tasa de aprendizaje alpha que es un hiper-parámetro dado antes de realizar el proceso iterativo (Bishop, 2006, p. 45).

<div aling = "center">

<img src="images/equationGD.png">

</div>

Esta implementacion se realiza en el siguiente código reciclado del cuaderno anteriormente mencionado y adaptado para los problemas posteriores (Zapata & Ospina, 2024).

```python
def mi_optimizador_num_dif_ndim(x0, g, lr=0.000001, tol=1e-10, max_iter=10000, h=1e-10, monitorCada=1000):
    # Inicialización de variables:
    # x0: punto inicial del vector de entrada en el espacio de soluciones
    # g: función objetivo que queremos minimizar
    # lr: tasa de aprendizaje (learning rate) que determina el tamaño del paso en cada iteración
    # tol: tolerancia para la convergencia del algoritmo
    # max_iter: número máximo de iteraciones permitido
    # h: paso pequeño para calcular la derivada numérica (gradiente)
    # monitorCada: cada cuántas iteraciones imprimir el progreso

    solucion = list()  # Lista para almacenar las soluciones obtenidas en cada iteración
    valor_obj = list()  # Lista para almacenar los valores de la función objetivo evaluados en cada solución
    k = 0  # Contador de iteraciones
    delta_sol = 1  # Diferencia entre soluciones sucesivas, usada para verificar la convergencia

    # Agregando valores iniciales
    # Agregando el valor de x inicial a la lista de soluciones
    solucion.append(x0)
    # Agregando el valor de la función objetivo evaluada en x0 a la lista de valores
    valor_obj.append(g(x0))

    # Mientras delta_sol sea mayor que la tolerancia (tol) y k sea menor que el número máximo de iteraciones (max_iter)
    while (delta_sol > tol) and (k < max_iter):
        # Actualización del punto x usando el gradiente numérico y la tasa de aprendizaje
        x_new = solucion[k] - lr * grad_num(solucion[k], g, h=h)
        # Agregamos la nueva solución a la lista de soluciones
        solucion.append(x_new)
        # Evaluamos la función objetivo en la nueva solución y la añadimos a la lista de valores
        valor_obj.append(g(x_new))
        # Incrementamos el contador de iteraciones
        k = k + 1
        # Calculamos la máxima diferencia absoluta entre la solución anterior y la nueva solución
        delta_sol = np.abs(solucion[k - 1] - solucion[k]).max()
        
        # Monitoreo del progreso cada cierto número de iteraciones
        if (k % monitorCada == 0):
            print(f"Iteración {k}/{max_iter}")  # Imprimimos el número de iteración actual
            # Monitoreo de los cambios en las soluciones
            # print(f"|sol[k-1]-sol[k]| {np.abs(solucion[k-1] - solucion[k])}")
            # Imprimimos la nueva solución y el valor de la función objetivo evaluada en ella
            print(f"X = {x_new}, f = {g(x_new)}\n")
    
    # Devolvemos las soluciones, los valores de la función objetivo y el número de iteraciones realizadas
    return solucion, valor_obj, k
```

### 1.2. Optimización por algorítmos genéticos.

Los algoritmos genéticos son una técnica de optimización motivada en los principios de la evolución biológica, por ejemplo: La selección natural, el cruce (crossover), la mutación, entre otras más (Geeks for Geeks, 2022).

Para realizar la implementación de esta idea, nos apoyaremos en la implementación que realiza la librería PyGad de este tipo de algoritmos, siguiendo los siguientes parámetros y carcaterísticas (Towards Data Science, 2020). Además del reciclado de algunos código del cuaderno anteriormente mencionado (Zapata & Opsina, 2024).

#### 1.2.1. Población inicial.

Definición de un conjunto de soluciones candidatas generadas aleatoriamente. Cada individuo en la población representa una posible solución al problema. Esto se relaciona con los siguientes parámetros que luego se implementará en una instancia de PyGad.

```python

init_range_low = -10.0
init_range_high = 10.0

```

#### 1.2.2. Evaluación por función fitness.

Cada individuo en la población se evalúa utilizando una función de aptitud para determinar qué tan buena es la solución. En nuestro caso, como se desea minimizar la función, la función fitness serán los valores negativos de su definición original, esto es, porque estos algoritmos buscan en esencia maximizar.

```python
def mi_f_fitness(ga_instance, solution, solution_idx):
    y = -f(solution)
    return y
```

#### 1.2.3. Selección.

Con base en su aptitud (Valor de la función fitness) se seleccionan individuos de la población actual para ser padres. Los individuos de la nueva generación reemplazan a los de la generación anterior. Además, la forma en que se realiza el reemplazo puede variar, con el tipo de selección "sss", el reemplazo se realiza de manera que algunos padres se mantienen.

```python
parent_selection_type = "sss"
keep_parents = 1
```

#### 1.2.4. Cruce.

El cruce simulando la recombinación genética, combina pares de padres para crear nuevos individuos. En este caso se implementa el tipo de cruce "single_point" (cruce de un solo punto), que toma un punto en el cromosoma y mezcla las partes de los padres para crear hijos.

```python
crossover_type = "single_point"
```

#### 1.2.5. Mutación.

La mutación otorga variabilidad a los individuos del problema al modificar aleatoriamente algunos genes, ayudando a explorar nuevas áreas del espacio de búsqueda.

```python
mutation_type = "random"
mutation_percent_genes = 10
```

Además de lo anterior, el algoritmo posee un número fijo de generaciones (iteraciones).
```python
num_generations = 50
```

#### 1.2.6. Creación de la instancia.

Dicha instancia, por la definición de parámetros, termina creando un objeto que está listo para correr y así, otorgar un conjunto de soluciones y valores de la función que logren minimizar la función.

```python
ga_instancia = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       gene_space = {'low': -1.0, 'high': 1.0},
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       save_solutions=True)
ga_instancia.run()
```

Obteniendo la mejor solución con `ga_instancia.best_solution()`: 

```python
solution, solution_fitness, solution_idx = ga_instancia.best_solution()
print("Mejor solución : {solution}".format(solution=solution))
print("Valor de la función objetivo = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Posición de la mejor solución = {solution_idx}".format(solution_idx=ga_instancia.solutions_fitness.index(solution_fitness)))
```

### 1.3. Función Goldstein-Price 

#### 2 Dimensiones.

La definición formal de la función Goldstein-Prince está dada por la siguiente regla (Surjanovic & Bingham, n.d.).

<img src="images/equationGp2d.png">

Dado el grado alto de polinomio de la expresión anterior, la función anterior se suele evaluar con el siguiente dominio: *xi* en *[-2,2]* para todo *i=1,2*.

De manera teórica, el mínimo global de la función está en *(x1',x_2')*=(0,-1), tal que *f(x1',x2')=-3*. Dicho valor, es la referencia a la hora de realizar los cálculos posteriores para la optimización de la función. En **figura 1** se puede observar las imagenes de dicha función.

<div aling="center">

**Figura 1. Superficie de la función GoldsteinPrice en su dominio.**

<img src="images/figGraficoGoldsteinPrice.png">

</div>

Con lo anterior, observamos que para valores cercanos a *(x1,x2)=(-2,2)*, la función tendrá a valores por encima de 400 mil, lo que indica que, para valores por fuera del dominio usual, tendrá a infinito. Por lo que, para el análisis nos conentraremos en aquel dominio.

Dependiendo donde se realice el gráfico, esta función presenta varios mínimo locales para puntos cercanos al del mínimo global. En **Figura 2** se observa la curva de nivel cercana a dicho punto global, en otras aproximaciones, se observará como varía dicha curva dependiendo del punto de enfoque

<div aling="center">

**Figura 2. Curva de nivel enfocada en la ubicación del mínimo global.**

<img src="images/figCurvaNivelGP2d.png">

</div>


## Referencias.

* Surjanovic, S., & Bingham, D. (n.d.). Goldstein-Price Function. Simon Fraser University. Recuperado el 29 de agosto de 2024, de https://www.sfu.ca/~ssurjano/goldpr.html.

* Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

* Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

* Zapata, & Ospina. (2024). IRNA202401_intro_optim.ipynb [Cuaderno de Python].

* Towards Data Science. (2020). Introduction to Genetic Algorithms — Including Example Code. Recuperado de https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3.

* Geeks for Geeks. (2022). Introduction to Genetic Algorithm. Recuperado de https://www.geeksforgeeks.org/introduction-to-genetic-algorithm/.
