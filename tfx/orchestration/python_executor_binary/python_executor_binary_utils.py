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
r"""Shared IR serialization logic used by TFleX python executor binary."""

import base64

from tfx.orchestration.portable import base_executor_operator
from tfx.proto.orchestration import executor_invocation_pb2


def deserialize_execution_info(
    execution_info_b64: str) -> base_executor_operator.ExecutionInfo:
  """De-serialize the ExecutionInfo class from a binary string."""
  exec_invocation_pb = executor_invocation_pb2.ExecutorInvocation.FromString(
      base64.b64decode(execution_info_b64))
  return base_executor_operator.ExecutionInfo.from_executor_invocation(
      exec_invocation_pb)


def serialize_execution_info(
    execution_info: base_executor_operator.ExecutionInfo) -> str:
  """Serialize the ExecutionInfo class from a binary string."""
  exec_invocation_pb = execution_info.to_executor_invocation()
  return base64.b64encode(
      exec_invocation_pb.SerializeToString()).decode('ascii')
