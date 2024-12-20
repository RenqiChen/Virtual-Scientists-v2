# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

from copy import deepcopy
from typing import Any, Dict, List

from camel.storages.key_value_storages import BaseKeyValueStorage


class InMemoryKeyValueStorage(BaseKeyValueStorage):
    r"""A concrete implementation of the :obj:`BaseKeyValueStorage` using
    in-memory list. Ideal for temporary storage purposes, as data will be lost
    when the program ends.
    """

    def __init__(self, max_memory_size) -> None:
        self.memory_list: List[Dict] = []
        self.max_memory_size = max_memory_size

    def save(self, records: List[Dict[str, Any]]) -> None:
        r"""Saves a batch of records to the key-value storage system.

        Args:
            records (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary represents a unique record to be stored.
        """
        self.memory_list.extend(deepcopy(records))
        if self.max_memory_size is not None and len(self.memory_list) > self.max_memory_size:
            self.memory_list = self.memory_list[-self.max_memory_size:]

    def load(self) -> List[Dict[str, Any]]:
        r"""Loads all stored records from the key-value storage system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                represents a stored record.
        """
        return deepcopy(self.memory_list)

    def clear(self) -> None:
        r"""Removes all records from the key-value storage system."""
        self.memory_list.clear()
