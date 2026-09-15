# Configuración Local

Este documento explica cómo podemos configur el entorno 
para ejecutar las tutoriales de forma local usando jupyter notebooks.

## Entorno Virtual

### Creando Entorno Virtual
Se recomienda utilizar virtual environment:

En una terminal, nos paramos en el directorio donde hayamos descomprimido los archivos y 
luego ejecutamos el comando:
```bash
pyhton3 -m venv .venv
```

Esto creará un directorio aislado
donde podremos instalar dependencias sin contaminar
la instalación de Python del sistema.

### Activando Entorno Virtual

Activamos el entorno de Python usando el comando:

#### En Linux / Mac 

```bash
# en Linux / Mac
source .venv/bin/activate
```

#### En Windows 
```bash
# en Windows
.venv\Scripts\activate
```

### Instalando e iniciando Jupyter Lab

Con el environment activado instalamos el paquete de jupyter
```bash
pip install jupyter
```

Esto podría demorar unos minutos.

Luego ejecutamos jupyter lab
```bash
jupyter lab
```
