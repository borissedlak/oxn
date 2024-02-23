"""Experiment runner"""
import logging
import random
import time
import uuid
import hashlib
import datetime

import psutil

import docker

from .errors import OxnException
from .treatments import (
    EmptyTreatment,
    NetworkDelayTreatment,
    PauseTreatment,
    PacketLossTreatment,
    PrometheusIntervalTreatment,
    StressTreatment,
    TailSamplingTreatment,
    KillTreatment,
    ProbabilisticSamplingTreatment,
    MetricsExportIntervalTreatment,
)
from . import utils
from .observer import Observer
from .pricing import Accountant
from .utils import utc_timestamp
from .models.treatment import Treatment

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Class that represents execution of experiments

    From the perspective of the runner, it always executes only a single run of an experiment.
    Multiple runs, asynchronous runs should be handled outside the runner.
    The runner builds the treatments, executes them in the provided order from the spec, observes the responses
    and then labels the resulting data with treatment information.
    The runner additionally waits for the specified time intervals depending on experiment configuration.

    """

    treatment_keys = {
        "kill": KillTreatment,
        "pause": PauseTreatment,
        "loss": PacketLossTreatment,
        "empty": EmptyTreatment,
        "delay": NetworkDelayTreatment,
        "stress": StressTreatment,
        "sampling": PrometheusIntervalTreatment,
        "tail": TailSamplingTreatment,
        "probabilistic_sampling": ProbabilisticSamplingTreatment,
        "metrics_interval":  MetricsExportIntervalTreatment,
    }

    def __init__(
        self,
        config=None,
        config_filename=None,
        additional_treatments=None,
        random_treatment_order=False,
        accountant_names=None,
    ):
        self.config = config
        """Experiment specification dict"""
        self.config_filename = config_filename
        """Experiment specification filename"""
        self.id = uuid.uuid4().hex
        """Random and unique ID to identify runs"""
        self.treatments = {}  # since python 3.6 dict remembers order of insertion
        """Treatments to execute for this run"""
        self.experiment_start = None
        """Experiment start as UTC unix timestamp in seconds"""
        self.experiment_end = None
        """Experiment end as UTC unix timestamp in seconds"""
        self.random_treatment_order = random_treatment_order
        """If the treatments should be executed in random order"""
        self.additional_treatments = (
            additional_treatments if additional_treatments else []
        )
        """Additional user-supplied treatments"""
        self.observer = Observer(config=self.config)
        """Observer for response variables"""
        self.accountant = None
        if accountant_names:
            self.accountant = Accountant(
                client=docker.from_env(),
                container_names=accountant_names,
                process=psutil.Process(),
            )
        """Accountant to determine resource expenditure during experiments"""
        self._compute_hash()
        """Compute unique identifiers for runs and experiment config"""
        self._extend_treatments()
        """Populate the treatment_keys class variable with additional user-supplied treatments"""
        self._build_treatments()
        """Populate the treatment dicts from the config and any user-supplied treatments"""

    def __repr__(self):
        return f"ExperimentRunner(config={self.config_filename}, hash={self.short_hash}, run={self.short_id})"

    @property
    def short_id(self) -> str:
        """Return the truncated run id for this experiment"""
        return self.id[:8]

    @property
    def short_hash(self) -> str:
        """Return the truncated hash for this experiment"""
        return self.hash[:8]

    @property
    def humanize_start_timestamp(self) -> datetime.datetime:
        """Return the human-readable start timestamp"""
        return utils.humanize_utc_timestamp(self.experiment_start)

    @property
    def humanize_end_timestamp(self) -> datetime.datetime:
        """Return the human-readable start timestamp"""
        return utils.humanize_utc_timestamp(self.experiment_end)

    def _compute_hash(self) -> None:
        """Hash the config filename to uniquely identify experiments"""
        if self.config_filename:
            self.hash = self.config_filename.encode("utf-8")
            self.hash = hashlib.sha256(usedforsecurity=False)
            self.hash = self.hash.hexdigest()

    def _build_treatments(self) -> None:
        """Build a representation of treatments defined in config"""
        treatment_section = self.config["experiment"]["treatments"]
        if self.random_treatment_order:
            random.shuffle(treatment_section)
        for treatment in treatment_section:
            key = next(iter(treatment))
            description = treatment[key]
            params = description["params"]
            action = description["action"]
            self.treatments[key] = self._build_treatment(
                action=action, params=params, name=key
            )
            logger.debug("Successfully built treatment %s", self.treatments[key])

    def _build_treatment(self, action, params, name) -> Treatment:
        """Build a single treatment from a description"""
        treatment_class = self.treatment_keys.get(action)
        try:
            instance = treatment_class(config=params, name=name)
        except TypeError:
            raise OxnException(
                message=f"Error while building treatment {name}",
                explanation=f"Treatment key {action} does not exist in the treatment library",
            )
        # TODO: move this out of the method. has to be called from the engine because the engine knows when services are ready
        if not instance.preconditions():
            raise OxnException(
                message=f"Error while checking preconditions for treatment {name}",
                explanation="\n".join(instance.messages),
            )
        return instance

    def _extend_treatments(self) -> None:
        """Extend the treatments the runner knows about with user-supplied treatments"""
        for treatment in self.additional_treatments:
            self.treatment_keys |= {treatment.action: treatment}


    # TODO: split into two methods, one that relies on container being started and one that modifies container infrastructure before theyre started
    def execute(self) -> None:
        """
        Execute one run of the experiment
        A single experiment run is defined as one execution of all treatments and one observation of all responses
        """
        self.experiment_start = utc_timestamp()
        self.observer.experiment_start = self.experiment_start
        if self.accountant:
            self.accountant.read_all_containers()
            self.accountant.read_oxn()
        ttw_left = self.observer.time_to_wait_left()
        logger.info(f"Sleeping for {ttw_left} seconds")
        time.sleep(ttw_left)
        logger.info(f"Starting treatments")
        for treatment in self.treatments.values():
            treatment.start = utc_timestamp()
            logger.debug(
                f"Treatment {treatment.name}: Start time {treatment.humanize_start_time}"
            )
            treatment.inject()
            if not hasattr(treatment.clean, "defer_cleanup"):
                treatment.clean()
                treatment.end = utc_timestamp()
                logger.debug(
                    f"Treatment {treatment.name}: End time {treatment.humanize_end_time}"
                )
            else:
                logger.debug(f"Deferred cleaning of treatment {treatment.name}")
        for treatment in self.treatments.values():
            if hasattr(treatment.clean, "defer_cleanup"):
                treatment.clean()
                treatment.end = utc_timestamp()
                logging.info(
                    f"Treatment {treatment.name}: End time {treatment.humanize_end_time}"
                )
        self.experiment_end = utc_timestamp()
        self.observer.experiment_end = self.experiment_end
        logger.info(f"Injected treatments")
        self.observer.initialize_variables()
        ttw_right = self.observer.time_to_wait_right()
        logger.info(f"Sleeping for {ttw_right} seconds")
        time.sleep(ttw_right)
        self.observer.observe()
        logger.info("Observed response variables")
        self._label()
        if self.accountant:
            self.accountant.read_all_containers()
            logger.debug(
                f"Read container resource data for {self.accountant.container_names}"
            )
            self.accountant.read_oxn()
            self.accountant.consolidate()

    def clear(self) -> None:
        """Clear the storage of the runner"""
        self.experiment_end = None
        self.experiment_start = None

    def _label(self) -> None:
        """Label the observed data with information from the treatments"""
        for treatment in self.treatments.values():
            for response_id, response_variable in self.observer.variables().items():
                response_variable.label(
                    treatment_end=treatment.end,
                    treatment_start=treatment.start,
                    label_column=treatment.name,
                    label=treatment.name,
                )
