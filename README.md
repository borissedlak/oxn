##### Installation
###### Setup the OpenTelemetry demo application
1.  Change to the forked demo submodule folder

```cd opentelemetry-demo/```

2. Run docker compose to start the demo

```docker compose up --no-build```

> Note: If you're on Apple Silicon, remove the ```--no-build``` flag to create local images. This might take up to 20 minutes.

3. Verify the demo application is working by visiting

* ```http:localhost:8080/``` for the Webstore
* ```http:localhost:8080/jaeger/ui``` for Jaeger
* ```http:localhost:9090``` for Prometheus

##### Install oxn (via pip)

1. Install virtualenv

```pip install virtualenv```

2. Create a virtualenv (named venv here)

```virtualenv venv```

3. Source the venv 

```source venv/bin/activate```

4. Install oxn requirements.

```pip install -r requirements.txt```

> Note: oxn requires the pytables package, which in turn requires a set of dependencies.
> If you are on macOS with an M1 chip, you probably have to install these dependencies via homebrew 
> 
> ```brew install hdf5 c-blosc lzo bzip2```
> 
> You then probably need to set the environment variables 
> 
> ```export HDF5_DIR=/opt/homebrew/opt/hdf5```
>
> ```export BLOSC_DIR=/opt/homebrew/opt/c-blosc```


##### Tests
> Note: Tests require the opentelemetry demo to be running and a working installation of coverage.py 

1.  Run tests with make

```make coverage```

2. View the coverage report 

```coverage report```


##### Run example observability experiments

1. You can run observability experiments from the example directory like so 

```python -m oxn examples/example_experiment.yml```

