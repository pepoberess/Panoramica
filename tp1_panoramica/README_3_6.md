# Ítem 3.6: cálculo del canvas

La explicación académica, el ejemplo ejecutado, las comprobaciones con `assert`
y el gráfico de las esquinas están en [tp1_pano.ipynb](tp1_pano.ipynb).
El notebook importa las funciones de este módulo y requiere NumPy, OpenCV
y Matplotlib, además de un entorno para ejecutar notebooks de Python.

## Uso e integración

Desde un notebook o script ubicado en la raíz del repo:

```python
from tp1_panoramica.panorama_bounds import compute_panorama_bounds

# Las imágenes y las dos matrices llegan de las etapas anteriores.
resultado = compute_panorama_bounds(
    img_left=img_izquierda,
    img_anchor=img_ancla,
    img_right=img_derecha,
    H_left_to_anchor=H_izq_ancla,
    H_right_to_anchor=H_der_ancla,
)

panorama_width, panorama_height = resultado["size"]
T = resultado["T"]
H_left_final = resultado["H_left_final"]
H_anchor_final = resultado["H_anchor_final"]
H_right_final = resultado["H_right_final"]
```

Si el notebook está dentro de `tp1_panoramica`, el import es
`from panorama_bounds import compute_panorama_bounds`.

Las matrices deben llevar puntos de izquierda/derecha **hacia el ancla**.
Usar las mismas imágenes (dimensiones y recortes) con las que se estimaron
las homografías. Si los compañeros devuelven también una máscara de inliers,
pasar únicamente la matriz 3x3. La función no depende de DLT ni RANSAC.

`size` ya tiene el orden `(ancho, alto)` requerido por OpenCV; las tres matrices
finales quedan listas para la futura etapa 3.7. Este módulo sólo calcula geometría:
no genera una imagen panorámica ni ejecuta warping o blending.

## Funciones y convención de bordes

- `transform_corners(image, H)` obtiene `h, w` de `image.shape[:2]`, construye
  `(0, 0), (w, 0), (w, h), (0, h)`, agrega la coordenada homogénea 1,
  multiplica por H y divide por la tercera coordenada. Devuelve un array
  `float64` de tamaño `(4, 2)`.
- `compute_panorama_bounds(...)` transforma las tres imágenes (identidad para
  el ancla), reúne las 12 esquinas, calcula límites, T y las matrices finales.
  También devuelve `bounds`, `corners_anchor` y `corners_final` para inspección.

Las esquinas representan **límites geométricos**, no índices de píxeles.
Una imagen de ancho w ocupa el intervalo de bordes `[0, w]`; sus columnas
se indexan de 0 a w-1. Por eso una esquina puede estar en x = ancho del canvas
sin quedar fuera de su borde geométrico. Lo mismo vale para el alto.

Para las 12 esquinas transformadas `(xi, yi)`:

```text
xmin = floor(min(xi))       xmax = ceil(max(xi))
ymin = floor(min(yi))       ymax = ceil(max(yi))
ancho = xmax - xmin         alto = ymax - ymin
```

No se suma 1: con identidad se recupera exactamente `(w, h)`.
`floor` y `ceil` redondean hacia afuera; el margen de redondeo es menor que
un píxel por lado en aritmética exacta. Es el rectángulo mínimo con límites
enteros en el sistema del ancla. Pueden quedar zonas vacías entre imágenes
inclinadas: quitar esas zonas exigiría recortar contenido.

## Por qué aparece T y por qué T @ H

Una coordenada negativa indica que la imagen se extiende hacia la izquierda
o arriba del origen del ancla. Se mueve todo el conjunto por igual con
`tx = -xmin`, `ty = -ymin`, preservando su posición relativa:

```text
T = [[1, 0, -xmin],
     [0, 1, -ymin],
     [0, 0,     1]]
```

Como se incluye el ancla con origen `(0, 0)`, los mínimos nunca son positivos.
Si no hay coordenadas negativas, T es la identidad.
Para un punto homogéneo p, primero se aplica H y después T:
`p_final = T @ (H @ p) = (T @ H) @ p`. Invertir ese orden trasladaría el punto
en el sistema original de la imagen. Para el ancla, `T @ I = T`.

Se valida que las imágenes sean válidas y no estén vacías, las matrices sean
3x3 y finitas, el denominador no sea cero, las coordenadas resultantes sean
finitas y el tamaño sea positivo. Se usa directamente la H recibida.
La comprobación de que las esquinas finales entran en el canvas queda en
los tests y en el notebook.

## Referencias locales

Se siguió `tutorial_codigos/homo_utils copy.py` (`homo`, `cart`,
`apply_transform`) y los notebooks `homografias copy.ipynb` (especialmente
la celda que calcula bordes y `translation_matrix @ H`) y
`transformaciones copy.ipynb`. Se adaptan las operaciones NumPy sin importar
el auxiliar, cuyo nombre contiene espacios y cuya importación requiere
Matplotlib. Se reemplaza el `round` del ejemplo por `floor`/`ceil`.
La consigna del TP, página 8, pide justificar tanto los límites como el ajuste.

## Pruebas

Requieren NumPy y OpenCV (`python3 -m pip install numpy opencv-python` si faltan).
Desde la raíz del repo:

```bash
python3 -B -m unittest discover -s tp1_panoramica -v
```

La prueba con `inputs/cuadro_1.jpg` usa la misma imagen en los tres roles y
traslaciones manuales de -200 y +200 en x. Para sus dimensiones `(4096, 3072)`:
los límites son `(-200, 0, 3272, 4096)`, T desplaza 200 en x y el canvas mide
`(3472, 4096)`. Las esquinas finales alcanzan los cuatro bordes.
También se prueban identidad, traslaciones fraccionarias para verificar
`floor`/`ceil`, una división proyectiva sencilla y entradas inválidas básicas.
En el notebook quedan ejecutadas las pruebas de identidad, traslaciones de
±200 y pertenencia de las esquinas al canvas. No se ejecuta el ítem 3.7.
