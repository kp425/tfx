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
"""Tests for tfx.orchestration.experimental.core.sync_pipeline_task_gen."""

import os

from absl.testing import parameterized
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.orchestration.portable import test_utils as tu
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


class SyncPipelineTaskGeneratorTest(tu.TfxTest, parameterized.TestCase):

  def setUp(self):
    super(SyncPipelineTaskGeneratorTest, self).setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._metadata_path = metadata_path
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    # Sets up the pipeline.
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata', 'sync_pipeline.pbtxt'),
        pipeline)
    self._pipeline = pipeline
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)
    self._pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        'test_run_0')

    # Extracts components.
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node

    self._task_queue = tq.TaskQueue()

  def _verify_node_execution_task(self, node, execution_id, task):
    self.assertEqual(
        task_lib.ExecNodeTask.create(self._pipeline, node, execution_id), task)

  def _dequeue_and_test(self, use_task_queue, node, execution_id):
    if use_task_queue:
      task = self._task_queue.dequeue()
      self._task_queue.task_done(task)
      self._verify_node_execution_task(node, execution_id, task)

  def _generate_and_test(self, use_task_queue, num_initial_executions,
                         num_tasks_generated, num_new_executions,
                         num_active_executions):
    """Generates tasks and tests the effects."""
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    self.assertLen(
        executions, num_initial_executions,
        'Expected {} execution(s) in MLMD.'.format(num_initial_executions))
    task_gen = sptg.SyncPipelineTaskGenerator(self._mlmd_connection,
                                              self._pipeline,
                                              self._task_queue.contains_task_id)
    tasks = task_gen.generate()
    self.assertLen(
        tasks, num_tasks_generated,
        'Expected {} task(s) to be generated.'.format(num_tasks_generated))
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    num_total_executions = num_initial_executions + num_new_executions
    self.assertLen(
        executions, num_total_executions,
        'Expected {} execution(s) in MLMD.'.format(num_total_executions))
    active_executions = [
        e for e in executions
        if e.last_known_state == metadata_store_pb2.Execution.RUNNING
    ]
    self.assertLen(
        active_executions, num_active_executions,
        'Expected {} active execution(s) in MLMD.'.format(
            num_active_executions))
    if use_task_queue:
      for task in tasks:
        self._task_queue.enqueue(task)
    return tasks, active_executions

  def _test_no_tasks_generated_when_new(self):
    task_gen = sptg.SyncPipelineTaskGenerator(self._mlmd_connection,
                                              self._pipeline, lambda _: False)
    tasks = task_gen.generate()
    self.assertEmpty(
        tasks,
        'Expected no task generation since ExampleGen is ignored for task '
        'generation and dependent downstream nodes are ready.')
    with self._mlmd_connection as m:
      self.assertEmpty(
          m.store.get_executions(),
          'There must not be any registered executions since no tasks were '
          'generated.')

  @parameterized.parameters(False, True)
  def test_tasks_generated_when_upstream_done(self, use_task_queue):
    """Tests that tasks are generated when upstream is done.

    Args:
      use_task_queue: If task queue is enabled, new tasks are only generated
        if a task with the same task_id does not already exist in the queue.
        `use_task_queue=False` is useful to test the case of task generation
        when task queue is empty (for eg: due to orchestrator restart).
    """
    # Simulate that ExampleGen has already completed successfully.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Before generation, there's 1 execution.
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    self.assertLen(executions, 1)

    # Generate once.
    with self.subTest(generate=1):
      tasks, active_executions = self._generate_and_test(
          use_task_queue,
          num_initial_executions=1,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1)
      self._verify_node_execution_task(self._transform, active_executions[0].id,
                                       tasks[0])

    # Should be fine to regenerate multiple times. There should be no new
    # effects.
    with self.subTest(generate=2):
      tasks, active_executions = self._generate_and_test(
          use_task_queue,
          num_initial_executions=2,
          num_tasks_generated=0 if use_task_queue else 1,
          num_new_executions=0,
          num_active_executions=1)
      if not use_task_queue:
        self._verify_node_execution_task(self._transform,
                                         active_executions[0].id, tasks[0])
    with self.subTest(generate=3):
      tasks, active_executions = self._generate_and_test(
          use_task_queue,
          num_initial_executions=2,
          num_tasks_generated=0 if use_task_queue else 1,
          num_new_executions=0,
          num_active_executions=1)
      execution_id = active_executions[0].id
      if not use_task_queue:
        self._verify_node_execution_task(self._transform, execution_id,
                                         tasks[0])

    # Mark transform execution complete.
    otu.fake_transform_output(self._mlmd_connection, self._transform,
                              active_executions[0])
    # Dequeue the corresponding task if task queue is enabled.
    self._dequeue_and_test(use_task_queue, self._transform, execution_id)

    # Trainer execution task should be generated when generate called again.
    with self.subTest(generate=4):
      tasks, active_executions = self._generate_and_test(
          use_task_queue,
          num_initial_executions=2,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1)
      execution_id = active_executions[0].id
      self._verify_node_execution_task(self._trainer, execution_id, tasks[0])

    # Mark the trainer execution complete.
    otu.fake_trainer_output(self._mlmd_connection, self._trainer,
                            active_executions[0])
    # Dequeue the corresponding task if task queue is enabled.
    self._dequeue_and_test(use_task_queue, self._trainer, execution_id)

    # No more components to execute so no tasks are generated.
    with self.subTest(generate=5):
      self._generate_and_test(
          use_task_queue,
          num_initial_executions=3,
          num_tasks_generated=0,
          num_new_executions=0,
          num_active_executions=0)
    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())


if __name__ == '__main__':
  tf.test.main()
