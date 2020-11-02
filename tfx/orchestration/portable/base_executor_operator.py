# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class to define how to operator an executor."""

import abc
from typing import Any, Dict, List, Optional

import attr
import six
from tfx import types
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import executor_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.utils import abc_utils

from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2


def _build_artifact_dict(proto_dict):
  """Build ExecutionInfo input/output artifact dicts."""
  result = {}
  for k, v in proto_dict.items():
    result[k] = []
    for artifact_struct in v.elements:
      if not artifact_struct.HasField('artifact'):
        raise RuntimeError('Only support artifact oneof field')
      artifact_and_type = artifact_struct.artifact
      result[k].append(
          artifact_utils.deserialize_artifact(artifact_and_type.type,
                                              artifact_and_type.artifact))
  return result


def _build_proto_artifact_dict(artifact_dict):
  """Build PythonExecutorExecutionInfo input/output artifact dicts."""
  result = {}
  for k, v in artifact_dict.items():
    artifact_list = metadata_store_service_pb2.ArtifactStructList()
    for artifact in v:
      artifact_struct = metadata_store_service_pb2.ArtifactStruct(
          artifact=metadata_store_service_pb2.ArtifactAndType(
              artifact=artifact.mlmd_artifact, type=artifact.artifact_type))
      artifact_list.elements.append(artifact_struct)
    result[k] = artifact_list
  return result


def _build_exec_property_dict(proto_dict):
  """Build ExecutionInfo.exec_properties."""
  result = {}
  for k, v in proto_dict.items():
    result[k] = getattr(v, v.WhichOneof('value'))
  return result


def _build_proto_exec_property_dict(exec_properties):
  """Build PythonExecutorExecutionInfo.execution_properties."""
  result = {}
  for k, v in exec_properties.items():
    value = metadata_store_pb2.Value()
    if isinstance(v, str):
      value.string_value = v
    elif isinstance(v, int):
      value.int_value = v
    elif isinstance(v, float):
      value.double_value = v
    else:
      raise RuntimeError('Unsupported type {} for key {}'.format(type(v), k))
    result[k] = value
  return result


# TODO(b/150979622): We should introduce an id that is not changed across
# retires of the same component run and pass it to executor operators for
# human-readability purpose.
# TODO(b/165359991): Restore 'auto_attribs=True' once we drop Python3.5 support.
@attr.s
class ExecutionInfo:
  """A struct to store information for an execution."""
  # LINT.IfChange
  # The metadata of this execution that is registered in MLMD.
  execution_metadata = attr.ib(type=metadata_store_pb2.Execution, default=None)
  # The input map to feed to executor
  input_dict = attr.ib(type=Dict[str, List[types.Artifact]], default=None)
  # The output map to feed to executor
  output_dict = attr.ib(type=Dict[str, List[types.Artifact]], default=None)
  # The exec_properties to feed to executor
  exec_properties = attr.ib(type=Dict[str, Any], default=None)
  # The uri to executor result, note that Executors and Launchers may not run
  # in the same process, so executors should use this uri to "return"
  # ExecutorOutput to the launcher.
  executor_output_uri = attr.ib(type=str, default=None)
  # Stateful working dir will be deterministic given pipeline, node and run_id.
  # The typical usecase is to restore long running executor's state after
  # eviction. For examples, a Trainer can use this directory to store
  # checkpoints.
  stateful_working_dir = attr.ib(type=str, default=None)
  # The config of this Node.
  pipeline_node = attr.ib(type=pipeline_pb2.PipelineNode, default=None)
  # The config of the pipeline that this node is running in.
  pipeline_info = attr.ib(type=pipeline_pb2.PipelineInfo, default=None)
  # LINT.ThenChange(../../proto/orchestration/executor_invocation.proto)

  def to_executor_invocation(
      self) -> executor_invocation_pb2.ExecutorInvocation:
    return executor_invocation_pb2.ExecutorInvocation(
        output_metadata_uri=self.executor_output_uri,
        stateful_working_dir=self.stateful_working_dir,
        execution_properties=_build_proto_exec_property_dict(
            self.exec_properties),
        input_dict=_build_proto_artifact_dict(self.input_dict),
        output_dict=_build_proto_artifact_dict(self.output_dict))

  @classmethod
  def from_executor_invocation(
      cls, executor_invocation: executor_invocation_pb2.ExecutorInvocation
  ) -> 'ExecutionInfo':
    return cls(
        executor_output_uri=executor_invocation.output_metadata_uri,
        stateful_working_dir=executor_invocation.stateful_working_dir,
        exec_properties=_build_exec_property_dict(
            executor_invocation.execution_properties),
        input_dict=_build_artifact_dict(executor_invocation.input_dict),
        output_dict=_build_artifact_dict(executor_invocation.output_dict))


class BaseExecutorOperator(six.with_metaclass(abc.ABCMeta, object)):
  """The base class of all executor operators."""

  SUPPORTED_EXECUTOR_SPEC_TYPE = abc_utils.abstract_property()
  SUPPORTED_PLATFORM_CONFIG_TYPE = abc_utils.abstract_property()

  def __init__(self,
               executor_spec: message.Message,
               platform_config: Optional[message.Message] = None):
    """Constructor.

    Args:
      executor_spec: The specification of how to initialize the executor.
      platform_config: The specification of how to allocate resource for the
        executor.

    Raises:
      RuntimeError: if the executor_spec or platform_config is not supported.
    """
    if not isinstance(executor_spec,
                      tuple(t for t in self.SUPPORTED_EXECUTOR_SPEC_TYPE)):
      raise RuntimeError('Executor spec not supported: %s' % executor_spec)
    if platform_config and not isinstance(
        platform_config, tuple(t for t in self.SUPPORTED_PLATFORM_CONFIG_TYPE)):
      raise RuntimeError('Platform spec not supported: %s' % platform_config)
    self._executor_spec = executor_spec
    self._platform_config = platform_config

  @abc.abstractmethod
  def run_executor(
      self,
      execution_info: ExecutionInfo,
  ) -> execution_result_pb2.ExecutorOutput:
    """Invokes the executor with inputs provided by the Launcher.

    Args:
      execution_info: A wrapper of the info needed by this execution.

    Returns:
      The output from executor.
    """
    pass
