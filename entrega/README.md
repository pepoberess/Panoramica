# TP1 - Panorámica

La entrega principal es `tp1_pano.ipynb`. El notebook contiene la explicación,
los experimentos y los resultados del pipeline completo: SIFT, A-NMS, matching,
DLT, RANSAC, cálculo del canvas, warping y blending.

## Estructura

```text
entrega/
├── imgs/                 # datasets Cuadro, UdeSA e imágenes propias
├── outputs/              # panorámicas finales
├── panorama/             # implementación auxiliar importada por el notebook
├── tests/                # pruebas unitarias
├── requirements.txt
└── tp1_pano.ipynb        # informe ejecutable y archivo a entregar
```

## Ejecución

Desde esta carpeta:

```bash
python3 -m pip install -r requirements.txt
jupyter lab tp1_pano.ipynb
```

En Jupyter, usar **Restart Kernel and Run All Cells**. Las rutas son relativas a
esta carpeta; no es necesario modificar el notebook. Las dos panorámicas finales
se vuelven a guardar en `outputs/`.

## Verificación opcional

```bash
python3 -m unittest discover -s tests -v
```

Las pruebas cubren DLT, RANSAC, reproyección, límites del canvas, máscaras y
blending.
