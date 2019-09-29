import logging
import time
import os
import json
import numpy as np
import threading
from six.moves import queue
from universe import rewarder, spaces, vectorized, pyprofile
from universe.utils import random_alphanumeric

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)


class Recording(vectorized.Wrapper):
    """
Record all action/observation/reward/info to a log file.

It will do nothing, unless given a (recording_dir='/path/to/results') argument.
recording_policy can be one of:
    'capped_cubic' will record a subset of episodes (those that are a perfect cube: 0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, and every multiple of 1000 thereafter).
    'always' records all
    'never' records none
recording_notes can be used to record hyperparameters in the log file

The format is line-separated json, with large observations stored separately in binary.

The universe-viewer project (http://github.com/openai/universe-viewer) provides a browser-based UI
for examining logs.

"""

    def __init__(self, env, recording_dir=None, recording_policy=None, recording_notes=None):
        super(Recording, self).__init__(env)
        self._log_n = None
        self._episode_ids = None
        self._step_ids = None
        self._episode_id_counter = 0
        self._env_semantics_autoreset = env.metadata.get('semantics.autoreset', False)
        self._env_semantics_async = env.metadata.get('semantics.async', False)
        self._async_write = self._env_semantics_async

        self._recording_dir = recording_dir
        if self._recording_dir is not None:
            if recording_policy == 'never' or recording_policy is False:
                self._recording_policy = lambda episode_id: False
            elif recording_policy == 'always' or recording_policy is True:
                self._recording_policy = lambda episode_id: True
            elif recording_policy == 'capped_cubic' or recording_policy is None:
                self._recording_policy = lambda episode_id: (int(round(episode_id ** (1. / 3))) ** 3 == episode_id) if episode_id < 1000 else episode_id % 1000 < 2
            else:
                self._recording_policy = recording_policy
        else:
            self._recording_policy = lambda episode_id: False
        logger.info('Running Recording wrapper with recording_dir=%s policy=%s. To change this, pass recording_dir="..." to env.configure.', self._recording_dir, recording_policy)

        self._recording_notes = {
            'env_id': env.spec.id,
            'env_metadata': env.metadata,
            'env_spec_tags': env.spec.tags,
            'env_semantics_async': self._env_semantics_async,
            'env_semantics_autoreset': self._env_semantics_autoreset,
        }
        if recording_notes is not None:
            self._recording_notes.update(recording_notes)

        if self._recording_dir is not None:
            os.makedirs(self._recording_dir, exist_ok=True)

        self._instance_id = random_alphanumeric(6)

    def _get_episode_id(self):
        ret = self._episode_id_counter
        self._episode_id_counter += 1
        return ret

    def _get_writer(self, i):
        """
        Returns a tuple of (log_fn, log_f, bin_fn, bin_f) to be written to by vectorized env channel i
        Or all Nones if recording is inactive on that channel
        """
        if self._recording_dir is None:
            return None
        if self._log_n is None:
            self._log_n = [None] * self.n
        if self._log_n[i] is None:
            self._log_n[i] = RecordingWriter(self._recording_dir, self._instance_id, i, async_write=self._async_write)
        return self._log_n[i]

    def _reset(self):
        if self._episode_ids is None:
            self._episode_ids = [None] * self.n
        if self._step_ids is None:
            self._step_ids = [None] * self.n

        for i in range(self.n):
            writer = self._get_writer(i)
            if writer is not None:
                if self._recording_notes is not None:
                    writer(type='notes', notes<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ChangeListManager">
    <list default="true" id="33c6d278-6d4a-4b2f-a65c-2e47f7c2a98e" name="Default Changelist" comment="" />
    <option name="EXCLUDED_CONVERTED_TO_IGNORED" value="true" />
    <option name="SHOW_DIALOG" value="false" />
    <option name="HIGHLIGHT_CONFLICTS" value="true" />
    <option name="HIGHLIGHT_NON_ACTIVE_CHANGELIST" value="false" />
    <option name="LAST_RESOLUTION" value="IGNORE" />
  </component>
  <component name="ProjectId" id="1Qp5qQRN48HCYGh9CCgbOlaB7LI" />
  <component name="ProjectLevelVcsManager" settingsEditedManually="true" />
  <component name="PropertiesComponent">
    <property name="last_opened_file_path" value="D:/Python" />
  </component>
  <component name="RunDashboard">
    <option name="ruleStates">
      <list>
        <RuleState>
          <option name="name" value="ConfigurationTypeDashboardGroupingRule" />
        </RuleState>
        <RuleState>
          <option name="name" value="StatusDashboardGroupingRule" />
        </RuleState>
      </list>
    </option>
  </component>
  <component name="RunManager" selected="Python.twisty">
    <configuration name="auth" type="PythonConfigurationType" factoryName="Python" temporary="true">
      <module name="AI GameBot using reinforcement learning" />
      <option name="INTERPRETER_OPTIONS" value="" />
      <option name="PARENT_ENVS" value="true" />
      <envs>
        <env name="PYTHONUNBUFFERED" value="1" />
      </envs>
      <option name="SDK_HOME" value="" />
      <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$/universe-master/universe/vncdriver" />
      <option name="IS_MODULE_SDK" value="true" />
      <option name="ADD_CONTENT_ROOTS" value="true" />
      <option name="ADD_SOURCE_ROOTS" value="true" />
      <option name="SCRIPT_NAME" value="$PROJECT_DIR$/universe-master/universe/vncdriver/auth.py" />
      <option name="PARAMETERS" value="" />
      <option name="SHOW_COMMAND_LINE" value="false" />
      <option name="EMULATE_TERMINAL" value="false" />
      <option name="MODULE_MODE" value="false" />
      <option name="REDIRECT_INPUT" value="false" />
      <option name="INPUT_FILE" value="" />
      <method v="2" />
    </configuration>
    <configuration name="demo" type="PythonConfigurationType" factoryName="Python" temporary="true">
      <module name="AI GameBot using reinforcement learning" />
      <option name="INTERPRETER_OPTIONS" value="" />
      <option name="PARENT_ENVS" value="true" />
      <envs>
        <env name="PYTHONUNBUFFERED" value="1" />
      </envs>
      <option name="SDK_HOME" value="" />
      <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
      <option name="IS_MODULE_SDK" value="true" />
      <option name="ADD_CONTENT_ROOTS" value="true" />
      <option name="ADD_SOURCE_ROOTS" value="true" />
      <option name="SCRIPT_NAME" value="$PROJECT_DIR$/demo.py" />
      <option name="PARAMETERS" value="" />
      <option name="SHOW_COMMAND_LINE" value="false" />
      <option name="EMULATE_TERMINAL" value="false" />
      <option name="MODULE_MODE" value="false" />
      <option name="REDIRECT_INPUT" value="false" />
      <option name="INPUT_FILE" value="" />
      <method v="2" />
    </configuration>
    <configuration name="twisty" type="PythonConfigurationType" factoryName="Python" temporary="true">
      <module name="AI GameBot using reinforcement learning" />
      <option name="INTERPRETER_OPTIONS" value="" />
      <option name="PARENT_ENVS" value="true" />
      <envs>
        <env name="PYTHONUNBUFFERED" value="1" />
      </envs>
      <option name="SDK_HOME" value="" />
      <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$/universe-master/universe" />
      <option name="IS_MODULE_SDK" value="true" />
      <option name="ADD_CONTENT_ROOTS" value="true" />
      <option name="ADD_SOURCE_ROOTS" value="true" />
      <option name="SCRIPT_NAME" value="$PROJECT_DIR$/universe-master/universe/twisty.py" />
      <option name="PARAMETERS" value="" />
      <option name="SHOW_COMMAND_LINE" value="false" />
      <option name="EMULATE_TERMINAL" value="false" />
      <option name="MODULE_MODE" value="false" />
      <option name="REDIRECT_INPUT" value="false" />
      <option name="INPUT_FILE" value="" />
      <method v="2" />
    </configuration>
    <recent_temporary>
      <list>
        <item itemvalue="Python.twisty" />
        <item itemvalue="Python.demo" />
        <item itemvalue="Python.auth" />
      </list>
    </recent_temporary>
  </component>
  <component name="Sv