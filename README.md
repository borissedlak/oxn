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

##### Install oxn via pip

> Note: oxn requires Python >= 3.10

1. Install virtualenv

```pip install virtualenv```

2. Create a virtualenv (named venv here)

```virtualenv venv```

3. Source the venv 

```source venv/bin/activate```

4. Install oxn

```pip install . ```

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


##### Run an example observability experiment

```
oxn --help
usage: oxn [-h] [--times TIMES] [--report REPORT] [--accounting] [--randomize] [--extend EXTEND] [--loglevel [{debug,info,warning,error,critical}]] [--logfile LOG_FILE] [--timeout TIMEOUT] spec

Observability experiments engine

positional arguments:
  spec                  Path to an oxn experiment specification to execute.

options:
  -h, --help            show this help message and exit
  --times TIMES         Run the experiment n times. Default is 1
  --report REPORT       Create an experiment report at the specified location. If the file exists, it will be overwritten. If it does not exist, it will be created.
  --accounting          Capture resource usage for oxn and the sue. Requires that the report option is set.Will increase the time it takes to run the experiment by about two seconds for each service in the sue.
  --randomize           Randomize the treatment execution order. Per default, treatments are executed in the order given in the experiment specification
  --extend EXTEND       Path to a treatment extension file. If specified, treatments in the file will be loaded into oxn.
  --loglevel [{debug,info,warning,error,critical}]
                        Set the log level. Choose between debug, info, warning, error, critical. Default is info
  --logfile LOG_FILE    Write logs to a file. If the file does not exist, it will be created.
  --timeout TIMEOUT     Timeout after which we stop trying to build the SUE. Default is 1m

```

1. You can run observability experiments from the example directory like so 

```oxn experiments/delay_experiment.yml```

2. After the experiment is completed, open an interactive python session

```
(venv) $ python                                                                                                                                                                                                                                                                                                                                   -- INSERT --
Python 3.11.2 (main, Feb 16 2023, 02:51:42) [Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
```

3. Import the trie to search for experiment keys by prefix
```
>>> from oxn.store import Trie, get_dataframe
>>> import pandas as pd
>>> t = Trie()

>>> keys = t.query(item="experiments/delay_experiment.yml")

['experiments/delay_experiment.yml/00c714c2/recommendation_service_traces']
```
4. Load the data from the store by key

```
>>> key = keys[0]
>>> df = get_dataframe(key=key)
>>> df[['span_id', 'trace_id', 'service_name', 'duration']]

                                           span_id                          trace_id           service_name  duration
start_time                                                                                                           
2023-04-04 11:24:53.515899+00:00  2b8c5bc4461b17de  8057f220111fad105e6273e32c78e839         frontend-proxy     28501
2023-04-04 11:24:53.531564+00:00  dde5ead716e7f3b2  8057f220111fad105e6273e32c78e839  productcatalogservice        15
2023-04-04 11:24:53.533753+00:00  053b68133b043002  8057f220111fad105e6273e32c78e839  productcatalogservice        10
2023-04-04 11:24:53.533761+00:00  a74404c60f4dec33  8057f220111fad105e6273e32c78e839  productcatalogservice         7
2023-04-04 11:24:53.533788+00:00  82b62972a9a99dd3  8057f220111fad105e6273e32c78e839  productcatalogservice         2
...                                            ...                               ...                    ...       ...
2023-04-04 11:26:14.810521+00:00  f8c75e0bd78dd9c0  602eeb0ab9151c51502b52fe0b98fa72         frontend-proxy     16036
2023-04-04 11:26:14.813754+00:00  56fe1d652eb7ed4d  602eeb0ab9151c51502b52fe0b98fa72  recommendationservice      9275
2023-04-04 11:26:14.823233+00:00  030330a39f62bde3  602eeb0ab9151c51502b52fe0b98fa72  recommendationservice       756
2023-04-04 11:26:14.813665+00:00  d57c26deb90b8f7c  602eeb0ab9151c51502b52fe0b98fa72  recommendationservice     10463
2023-04-04 11:26:14.813701+00:00  869778e93a1da514  602eeb0ab9151c51502b52fe0b98fa72  recommendationservice     10329

```
5. Check that the data is labelled
```
>>> df.short_delay_treatment.describe()
count           16775
unique              2
top       NoTreatment
freq            15134
```
6. Confirm that the treatment had an effect on the response
```
>>> df.groupby('short_delay_treatment').duration.describe()
                                                      count          mean            std  min    25%     50%       75%       max
short_delay_treatment                                                                                                           
NetworkDelayTreatment(name=short_delay_treatmen...   1641.0  81255.688605  119798.933402  2.0  405.0  1196.0  209583.0  338446.0
NoTreatment                                         15134.0   5129.168561    7268.584810  2.0   16.0   902.0    9675.0  237092.0
```
7. Clearly, the mean span duration is increased in the treatment group.






