## Introduction
This artifact presents the tool OXN - **O**bservability e**X**periment e**N**gine. 
OXN is an extensible software framework to run observability experiments and compare observability design decisions.
OXN follows the design principles of cloud benchmarking and strives towards portable and repeatable experiments.
Experiments are defined as yaml-based configuration files, which allows them to be shared, versioned and repeated.
OXN automates every step of the experiment process in a straightforward manner, from SUE setup to data collection, processing and reporting. 


## Installation

##### Prerequisites
- Docker + Docker Compose
- Python >= v3.10
- Jupyter


###### Setup the OpenTelemetry demo application
1.  Change to the forked demo submodule folder

    ```cd opentelemetry-demo/```

2. Build needed containers. This will take a while a while

    ``` make build ```

    Alternativly, you can just build the container with fault injection, e.g., the recommender service. This may cause incompatability in the future. 

    ``` docker compose build recommendationservice ```

3. Run docker compose to start the demo

    ```docker compose up```

3. Verify the demo application is working by visiting

#TODO: Check what metrics there are
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

## Reproducing results of the paper
The experiments presented in the paper take a significant time to reproduce. 
While we included all the necessary experiments in the [experiments directory](/experiments), we provide the procedure for the PacketLoss-Experiment as an example here.
> Note: These results refer to section VI-B and VI-C of the paper 

1. You can run the packetloss experiment for the baseline configuration from the experiments directory as follows:

    ```oxn experiments/recommendation_loss15_baseline.yml --report recommendation_loss15_baseline.yaml```

    ```log
    [2024-02-28 07:18:10,907] isca/INFO/oxn.engine: Running experiment experiments/recommendation_loss15_baseline.yml for 1 times
    [2024-02-28 07:18:10,907] isca/INFO/oxn.engine: Experiment run 1 of 1
    [2024-02-28 07:18:11,409] isca/INFO/oxn.runner: Starting compile time treatments
    [2024-02-28 07:18:40,865] isca/INFO/oxn.engine: Started sue
    [2024-02-28 07:18:40,959] isca/INFO/oxn.treatments: Probed container recommendation-service for tc with result 0
    [2024-02-28 07:18:40,959] isca/INFO/locust.runners: Shape test starting.
    [2024-02-28 07:18:40,960] isca/INFO/oxn.engine: Started load generation
    [2024-02-28 07:18:40,960] isca/INFO/oxn.runner: Sleeping for 240.0 seconds
    [2024-02-28 07:18:40,961] isca/INFO/locust.runners: Shape worker starting
    [2024-02-28 07:18:40,961] isca/INFO/locust.runners: Shape test updating to 50 users at 25.00 spawn rate
    [2024-02-28 07:18:40,962] isca/INFO/locust.runners: Ramping to 50 users at a rate of 25.00 per second
    [2024-02-28 07:18:41,968] isca/INFO/locust.runners: All users spawned: {"CustomLocust": 50} (50 total users)
    [2024-02-28 07:22:40,961] isca/INFO/oxn.runner: Starting runtime treatments
    [2024-02-28 07:24:41,240] isca/INFO/oxn.treatments: Cleaned packet loss treatment in container recommendation-service.
    [2024-02-28 07:24:41,242] isca/INFO/oxn.runner: Injected treatments
    [2024-02-28 07:24:41,242] isca/INFO/oxn.runner: Cleaning compile time treatments
    [2024-02-28 07:24:41,243] isca/INFO/oxn.runner: Sleeping for 240.0 seconds
    [2024-02-28 07:28:48,431] isca/INFO/oxn.runner: Observed response variables
    [2024-02-28 07:28:48,438] isca/INFO/oxn.engine: Stopped load generation
    [2024-02-28 07:28:48,487] isca/INFO/oxn.engine: Wrote frontend_traces to store
    [2024-02-28 07:28:48,526] isca/INFO/oxn.engine: Wrote recommendation_traces to store
    [2024-02-28 07:28:48,548] isca/INFO/oxn.engine: Wrote system_CPU to store
    [2024-02-28 07:28:48,566] isca/INFO/oxn.engine: Wrote recommendations_total to store
    [2024-02-28 07:29:36,256] isca/INFO/oxn.engine: Stopped sue
    [2024-02-28 07:29:36,256] isca/INFO/oxn.engine: Experiment run 1 of 1 completed
    ```

2. You can conduct further experiments to evaluate observability design alternatives A, B and C:

    ```oxn experiments/recommendation_loss15_A.yml --report recommendation_loss15_A.yaml```

    ```oxn experiments/recommendation_loss15_B.yml --report recommendation_loss15_B.yaml```

    ```oxn experiments/recommendation_loss15_C.yml --report recommendation_loss15_C.yaml```


3. After the experiments are completed, open the [Evaluation Notebook](Evaluation.ipynb) 

4. Execute step [1] of notebook

5. Jump to step [5] of notebook , and follow the instructions.

>Note: Plots and visibility scores may differ from paper due to:
  > - varying performance of machines
  > - changes in the [demo application](https://github.com/open-telemetry/opentelemetry-demo) that have occured since writing the paper
  > - changes in the OpenTelemetry tooling since the release of the paper